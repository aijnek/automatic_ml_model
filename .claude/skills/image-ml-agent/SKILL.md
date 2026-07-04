---
name: image-ml-agent
description: 画像を入力とする予測モデル（分類・回帰）をエンドツーエンドで自動構築するMLエンジニアリングエージェント。「〜を判定する画像分類モデルを作りたい」「写真から〜を予測したい」「画像を集めてモデルを学習して」「データ収集からアノテーション・学習まで自動で」等の依頼で使用。画像収集→Claude自身によるアノテーション→config生成→自律学習パイプライン起動→結果報告まで行う。
---

# image-ml-agent: 画像MLモデルのE2E自動構築

ユーザーの「画像に関する課題」を入力に、学習画像の収集 → アノテーション →
自律学習パイプラインの起動 → 結果報告までを一気通貫で行う。

## 全体フロー

| Phase | 内容 | 詳細 |
|---|---|---|
| 0 | 問題定義・ワークスペース確認・config生成 | 下記 |
| 1 | 収集計画の設計 → **チェックポイント①** | [references/collection.md](references/collection.md) |
| 2 | 収集実行・品質スクリーニング | 同上 |
| 3 | ルーブリック作成・サンプルアノテ → **チェックポイント②** | [references/annotation.md](references/annotation.md) |
| 4 | 全量アノテーション（サブエージェント並列）・マージ | 同上 |
| 5 | プリフライト・split・run_loop起動 | [references/pipeline.md](references/pipeline.md) |
| 6 | 監視・最終報告 | 同上 |

## 絶対ルール

1. **チェックポイント①②では必ずユーザーの承認を待つ**。承認前に次のPhaseへ進まない
   - ①: クラス設計・収集バケット・クエリ・目標枚数を提示してから収集
   - ②: サンプルアノテーション結果（10〜15枚の判定表）を提示してから全量アノテ
2. **収集バケット ≠ ラベル**。バケット名から機械的にラベルを付けてはならない。
   全画像をルーブリックに基づき独立に判定する（最大のラベルノイズ源）
3. **既存の成果物を黙って消さない・使い回さない**。Phase 0 のワークスペース確認に従う
4. **test split には触れない**。評価は run_loop が最後に1回だけ行う

## Phase 0 — 問題定義・ワークスペース確認・config生成

1. ユーザーから聞き取る（不足分のみ質問）:
   - タスク種別: classification / regression
   - 何を予測するか（description — 特徴量設計LLMへの指示になるので具体的に）
   - クラス名（分類）/ 値域と単位（回帰）
   - 合格ライン threshold（デフォルト 0.9。動作確認目的なら 0.5 を提案）
   - 収集規模（デフォルト: クラスあたり約50枚）
2. ワークスペース確認: `data/images/`・`data/annotations.csv`・`data/splits/`・
   `state.json`・`schemas/`・`features/`・`models/`・`results/` の存在を調べる。
   **別課題の成果物が残っていたら** `archive/<課題slug>_<YYYYMMDD>/` への退避（mv・可逆）を
   提案し、承認を得てから実行する
3. config.yaml を生成する（[references/pipeline.md](references/pipeline.md) のワンライナー。
   dataclass 経由必須）。以降の rubric・マージ検証がこれを参照する

## Phase 1–2 — 収集

[references/collection.md](references/collection.md) に従い `data/collection_plan.json` を設計し、
チェックポイント①で承認を得てから:

```bash
uv run python scripts/collect_images.py --plan data/collection_plan.json
```

バックグラウンドで実行し進捗をポーリング。完了後にバケット毎の到達数を確認し、
70%未満のバケットはクエリ拡張ループ（最大2ラウンド）。各バケット約5枚を Read で
目視スポットチェック（クリップアート・被写体違いの混入検知）。

## Phase 3–4 — アノテーション

[references/annotation.md](references/annotation.md) に従う:

1. `data/annotation/rubric.md` を書く（クラス定義・境界ルール・DISCARD基準）
2. 10〜15枚をメインセッションで判定 → 表で提示 → **チェックポイント②**
   （フィードバックで rubric を更新してから先へ）
3. 残り全量を25枚ずつのバッチでサブエージェント2〜4並列に分担
   （出力: `data/annotation/batches/batch_NN.csv`。冪等: 既存バッチのfilenameは除外）
4. マージ: `uv run python .claude/skills/image-ml-agent/scripts/merge_annotations.py`
   → `data/annotations.csv` 生成。クラス不均衡チェック（全クラス ≥ 14枚が下限）

## Phase 5–6 — 学習と報告

[references/pipeline.md](references/pipeline.md) に従う: `ollama list` でモデル存在確認 →
`uv run python -m pipeline.split` → `uv run python -m pipeline.run_loop` を
バックグラウンド起動 → `state.json` と `results/loop.log` を数分間隔で監視 →
完了後 `results/final_report.md` を要約し、データセット統計・ラベル品質注記・
ライセンス内訳を添えて報告する。

## 再開検出（スキル起動時に必ずチェック）

途中で中断されたセッションの続きを頼まれることがある。以下の順で状態を判定し、
該当Phaseから再開する（ユーザーに現状認識を示してから）:

| 見つかったもの | 再開位置 |
|---|---|
| `state.json`（finished=false） | Phase 6（run_loop 再起動・監視） |
| `data/annotations.csv` | Phase 5 |
| `data/annotation/batches/*.csv` が画像の一部のみカバー | Phase 4（未処理分のみ） |
| `data/annotation/rubric.md` | Phase 3〜4 |
| `data/collection_plan.json` + バケット目標達成 | Phase 3 |
| `data/collection_plan.json` のみ | Phase 2 |
| いずれもなし | Phase 0 |

## 制約・注意

- 依存管理は uv（`uv run` / `uv add`）。ollama がローカルで動いていることが学習の前提
- 収集ソースはキーレス（openverse / wikimedia）だけでも動く。.env にAPIキーがあれば自動追加
- 正解ラベルはClaude視覚アノテーション由来。その精度がモデル性能の上限になることを
  最終報告に必ず明記する
