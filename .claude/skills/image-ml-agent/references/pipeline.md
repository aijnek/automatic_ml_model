# パイプライン: config生成・起動・監視・報告

## config.yaml の生成（Phase 0）

必ず `pipeline.config` の dataclass 経由で生成する（スキーマ保証）。手書きYAMLは禁止。

```bash
uv run python - <<'EOF'
from pipeline.config import Config, save_config
save_config(Config(
    task_type="classification",          # または "regression"
    description="写真に犬が写っているかを判定する",  # 特徴量設計LLMへの指示になるので具体的に
    classes=["dog", "no_dog"],           # 分類のみ
    # target_min=0.0, target_max=5.0, target_unit="点",  # 回帰のみ
    threshold=0.9,                       # val macro-F1 / R² の合格ライン
))
EOF
```

指定しなかったフィールド（llm設定・verification・feature_selection）は妥当なデフォルトが入る。
ユーザーが希望した場合のみ上書きする。主なノブ:

| フィールド | デフォルト | 用途 |
|---|---|---|
| `threshold` | 0.9 | 合格ライン。下げるとループが早く終わる |
| `vlm_model` | `qwen3.5:9b` | 特徴量抽出VLM（ollamaモデル名） |
| `verification_enabled` | True | 抽出品質検証（step 4.5）。スモークランでは False |
| `feature_selection_enabled` | True | 合格後の特徴量後方消去（step 7.5） |

**スモークラン設定**（動作確認だけしたいとき）: `threshold=0.5, verification_enabled=False`

## プリフライト（Phase 5、起動前に必ず）

```bash
ollama list
```

- `vlm_model`（既定 `qwen3.5:9b`）が一覧にあるか
- `verification_enabled` なら `reference_vlm_model`（既定 `qwen3.6:27b-mlx`）もあるか
- designer は `ollama launch claude --model qwen3.6:35b-a3b-coding-nvfp4 -- -p` を subprocess 実行する。
  このモデルもあるか確認

無いモデルがあればユーザーに報告して `ollama pull` するか、config の該当フィールドを
手持ちモデルに変える（勝手にpullしない — サイズが大きい）。

## 分割と学習ループの起動

```bash
uv run python -m pipeline.split           # data/splits/{train,val,test}.csv 生成
uv run python -m pipeline.run_loop        # 自律ループ（合格まで無制限）
```

- `pipeline.split` は **annotations.csv が10件未満だとエラー**。既存 splits がある場合は
  上書きせずエラーになる — `--force` は「学習データの分け直し = 過去の学習結果が無効になる」
  ことをユーザーに明示して確認を得てから
- `run_loop` は長時間かかる（1イテレーション = designer LLM + 全画像VLM抽出 + 学習）。
  **バックグラウンドBashで起動**し、ポーリングで監視する
- `--max-iterations N` で N イテレーション後に一旦停止できる（再実行で継続）

## 監視（Phase 6）

- **ログ**: `results/loop.log`（stdoutと同内容）。tail で進捗確認
- **状態**: `state.json`
  - `iteration`: 現在のイテレーション番号
  - `best`: `{iteration, val_score}` これまでのベスト
  - `history`: 各イテレーションの記録
  - `finished`: true になったら完了
  - `test_score` / `feature_selection`: 完了後に入る
- ポーリング間隔は数分単位で十分（1イテレーションが数十分かかる）
- val スコアが数イテレーション停滞している場合は途中経過をユーザーに報告
  （`results/iter{N}/report.md` に診断がある）

## 中断と再開

すべて再開可能。セッションを跨ぐ場合はユーザーに以下を伝える:

| 状況 | 再開方法 |
|---|---|
| run_loop が途中 | `uv run python -m pipeline.run_loop`（state.json から続行） |
| 抽出が途中 | 同上（features CSV に追記済み分はスキップされる） |
| 最初からやり直したい | `state.json` を削除して run_loop 再実行 |

## 最終報告（Phase 6）

`results/final_report.md`（テスト評価・スキーマ変遷）を要約し、以下を加えて報告する:

1. **テストスコア**と合格ライン、イテレーション数
2. **採用された特徴量**（feature_selection 後のスキーマ。`schemas/metadata_v{N}_selected.json`）
3. **データセット統計**: 収集枚数 / DISCARD数と主な理由 / クラス分布
4. **ラベル品質の注記**: 正解ラベルはClaude視覚アノテーション由来であり、その精度が
   モデル性能の上限になる旨
5. **ライセンス内訳**: `collection_metadata.csv` の license 列を集計。NC/ND 混入があれば
   商用利用への注意を明記

## 成果物の場所

| パス | 内容 |
|---|---|
| `config.yaml` | タスク定義・学習設定 |
| `data/annotations.csv` | 正解ラベル |
| `data/splits/` | train/val/test 分割 |
| `schemas/metadata_v{N}.json` | 特徴量スキーマ（イテレーション毎） |
| `features/features_v{N}.csv` | 抽出済み特徴量 |
| `models/model_v{N}.txt` | LightGBMモデル |
| `results/iter{N}/report.md` | イテレーション診断 |
| `results/final_report.md` | 最終レポート |
