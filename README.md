# automatic-ml-model

画像からマルチモーダルLLM（VLM）でメタデータ（特徴量）を抽出し、LightGBM で分類/回帰モデルを作る自律開発パイプライン。

特徴量の設計（項目とVLMへのプロンプト）はローカルLLMが行い、val スコアが合格ラインを超えるまで **人間の介入なしに** 「設計 → 抽出 → 学習 → 評価 → 再設計」を繰り返す。

```
① 問題設定・アノテーション（Streamlit / 人間 — または image-ml-agent スキルで自動）
        ↓
② train/val/test split（1回だけ・以降固定）
        ↓
┌─ ③ 特徴量スキーマ設計/改訂  … ollama launch claude (qwen3.6:35b-a3b-coding-nvfp4)
│       ↓
│  ③.5 抽出品質検証           … 新規/変更特徴量のみ・trainサンプルで自己一致率＋参照VLM一致率
│       ↓
│  ④ メタデータ抽出           … ollama chat + structured outputs (qwen3.5:9b)
│       ↓
│  ⑤ LightGBM 学習・val評価
│       ↓
└─ ⑥ val < 合格ライン なら ③ へ（診断レポート＋抽出信頼性をフィードバック）
        ↓ val ≥ 合格ライン
⑦ test で最終評価 → results/final_report.md
```

## 前提

- [uv](https://docs.astral.sh/uv/)、[ollama](https://ollama.com)（`qwen3.6:35b-a3b-coding-nvfp4` と `qwen3.5:9b` をDL済み）
- macOS では LightGBM 用に `brew install libomp`

## 使い方

### 全自動（image-ml-agent スキル）

Claude Code から「〜を判定する画像分類モデルを作りたい」「写真から〜を予測したい」などと
依頼すると、`.claude/skills/image-ml-agent/` のプロジェクトスキルが起動し、
**画像収集 → Claude自身の視覚によるアノテーション → config生成 → 下記②③以降の
自律ループ起動 → 結果報告** までを一気通貫で行う。人間の確認は2箇所だけ
（①収集計画の承認、②サンプルアノテーション品質の承認）。詳細は
[.claude/skills/image-ml-agent/SKILL.md](.claude/skills/image-ml-agent/SKILL.md)。

### 手動（ステップごとに実行）

```bash
# 0. （任意）学習用画像の収集。複数の画像APIから8カテゴリ×50枚を data/images/ に収集し、
#    出典・ライセンスを data/collection_metadata.csv に記録する。不足カテゴリは画像加工で補完
uv run python scripts/collect_images.py                      # キー設定済みの全ソース
uv run python scripts/collect_images.py --sources wikimedia  # ソースを限定
uv run python scripts/collect_images.py --plan plan.json     # 任意課題のカテゴリ/クエリをJSONで指定
#    （--plan のJSON形式は scripts/collect_images.py の docstring 参照。組み込み8カテゴリの代わりに
#      queries / keywords / synthetic / per_category をカテゴリごとに定義できる）

# 0'. （任意）目視で選別しながら収集するUI。ソースとクエリを選んで検索 → サムネイル
#     一覧から適した画像だけチェックして保存（保存済み・除外済みはグレーアウト/非表示）
uv run streamlit run app/collect.py

# 1. 問題設定 + 画像アップロード + アノテーション（ブラウザが開く）
uv run streamlit run app/annotate.py

# 2. train/val/test 分割（70/15/15）
uv run python -m pipeline.split

# 3. 自律改善ループ開始（以降は完全無人。Ctrl+C で中断→再実行で再開）
uv run python -m pipeline.run_loop
```

完了すると `results/final_report.md` に test スコア・混同行列・特徴量スキーマの変遷が出力される。

### 画像収集ソースとAPIキー

| ソース | APIキー | 取得先 | レート制限 | 記録されるライセンス |
|---|---|---|---|---|
| openverse | 不要 | — | 匿名は低め（検索間3秒） | CC (`by-nc-nd` 等) |
| wikimedia | 不要 | — | 検索間1.5秒 | `CC BY-SA 4.0` 等 |
| pixabay | `PIXABAY_API_KEY` | https://pixabay.com/api/docs/ | 100 req/分 | `pixabay` (Content License) |
| pexels | `PEXELS_API_KEY` | https://www.pexels.com/api/ | 200 req/時 | `pexels` (Pexels License) |
| flickr | `FLICKR_API_KEY` ※ | https://www.flickr.com/services/api/ | 3600 req/時 | CC (`by` 等、CC系のみ検索) |
| google | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | https://developers.google.com/custom-search | 100 クエリ/日（無料枠） | `unknown` ⚠️ |

キーはプロジェクトルートの `.env` に書く（gitignore 済み）:

```bash
PIXABAY_API_KEY=xxxx
PEXELS_API_KEY=xxxx
FLICKR_API_KEY=xxxx
```

- ※ Flickr の新規APIキー発行は有料の Flickr Pro アカウント限定。ただし Flickr の CC 画像は Openverse がインデックスしているため、openverse ソース経由でほぼカバーできる
- `--sources auto`（既定）はキー設定済みのソースを全部使い、複数ソースをインターリーブして収集する。キー未設定のソースは `[skip]` 表示でスキップされる
- google はライセンス情報が取得できず `license=unknown` で記録されるため、既定には含まれない。`--sources google` と明示指定したときだけ使う最終手段
- 重複は「ソースID・掲載元URL・画像バイトのMD5」の3段階で排除される（Openverse経由で取得済みのFlickr写真をFlickr直APIで再取得しない）

### 動作確認（合成データでのE2E）

```bash
uv run python scripts/make_dummy_dataset.py --n 60   # 色付き図形60枚 + config + アノテーション生成
uv run python -m pipeline.split
uv run python -m pipeline.run_loop
```

### テスト

```bash
uv run pytest   # LLM呼び出しは全てモック。実LLM不要で数秒で完走
```

## 成果物の配置

| パス | 内容 |
|---|---|
| `config.yaml` | 問題設定（タスク種別・クラス・合格ライン・使用モデル） |
| `data/annotations.csv` | アノテーション結果 |
| `data/collection_plan.json` | 収集プラン（image-ml-agent スキル使用時） |
| `data/annotation/` | ルーブリックとアノテーションバッチCSV（同上） |
| `data/splits/` | train/val/test 分割（`--force` なしでは上書き不可） |
| `schemas/metadata_v{N}.json` | イテレーションNの特徴量スキーマ（項目・型・VLMプロンプト） |
| `features/features_v{N}.csv` | 抽出済みメタデータ（差分抽出のキャッシュ兼用） |
| `models/model_v{N}.txt` | LightGBM モデル |
| `schemas/metadata_v{N}_selected.json` | 特徴量選択後のスキーマ（合格イテレーションのみ） |
| `models/model_v{N}_selected.txt` | 特徴量選択後のモデル（除去が1件以上あった場合のみ） |
| `results/iter{N}/extraction_quality.json` | 抽出品質検証の結果（特徴量ごとの信頼性シグナル） |
| `results/iter{N}/report.md` | 診断レポート（designerへのフィードバック） |
| `results/feature_selection.json` | 特徴量選択の履歴（ラウンドごとの採用/棄却） |
| `results/final_report.md` | 最終レポート（test評価） |
| `state.json` | ループ進行状態（削除すると最初からやり直し） |

## 設計メモ

- **合格判定**: 分類 = val macro-F1、回帰 = val R²。閾値は `config.yaml` の `target_metric_threshold`（既定 0.9）。達成まで無制限にループする。
- **test リーク防止**: test は最終評価の1回しか使わない。split の再生成には `--force` が必要。
- **差分抽出**: 特徴量の再抽出要否は designer の申告（action）ではなく、定義（type/choices/prompt）の同一性で判定する。変更のない特徴量は前バージョンのCSVから流用。
- **再開性**: スキーマ・特徴量CSV（1画像ごと追記）・state.json が全てディスクに残るため、どのフェーズで中断しても再実行で続きから走る。
- **障害耐性**: designer の不正JSONはエラー内容を添えて最大3回リトライ。VLM抽出の失敗はリトライ後 NaN（LightGBM が NaN を扱える）。
- **抽出品質検証（③.5）**: 正解ラベルなしで特徴量ごとの抽出信頼性を測り、designer が「プロンプトを直すべき特徴量」と「削除すべき特徴量」を区別できるようにする。
  - **自己一致率**: trainからのサンプル画像（既定25枚）を temperature>0 で複数回（既定3回）再抽出したときの一致率。低い＝プロンプトが曖昧
  - **参照VLM一致率**: 大型の参照VLM（既定 `qwen3.6:27b-mlx`。未DLなら自動スキップ）の抽出結果との一致率。低い＝安定してブレないが間違っている可能性
  - **NaN率・最頻値割合**（全データ・追加コストなし）: 抽出失敗と縮退（全画像ほぼ同じ値＝情報量なし）の検出
  - 判定（`安定`/`不安定`/`縮退`/`抽出失敗`）は診断レポートの「抽出信頼性」セクションとして designer に渡り、改訂方針（プロンプト書き直し/削除/質問の単純化）に反映される
  - 対象は新規/変更特徴量のみ（差分方式）。スキーマ収束後はコストゼロ。`config.yaml` の `verification:` ブロックで無効化・調整可能:

```yaml
verification:
  enabled: true
  reference_vlm_model: qwen3.6:27b-mlx
  sample_size: 25
  n_repeats: 3
  temperature: 0.7
  vlm_timeout: 300
  thresholds:
    consistency: 0.7      # 自己一致率がこれ未満 → 不安定
    cross_agreement: 0.6  # 参照VLM一致率がこれ未満 → 不安定
    nan_rate: 0.2         # NaN率がこれ超 → 抽出失敗
    mode_fraction: 0.95   # 最頻値割合がこれ超 → 縮退
```

- **特徴量選択（⑦.5）**: val 合格後・test 評価前に、gain 重要度が最小の特徴量を1つずつ除去→再学習する後方消去を自動実行する。合格モデルの val スコア（baseline）からの低下が `max_score_drop` 以内なら採用、超えたら棄却して終了。選択中に test は一切使わず、最終評価は選択後のスキーマ・モデルで1回だけ行う。`config.yaml` の `feature_selection:` ブロックで無効化・調整可能:

```yaml
feature_selection:
  enabled: true         # false でモデル学習のみ（従来どおり）
  max_score_drop: 0.01  # baseline からの許容 val スコア低下
  min_features: 1       # これ未満には削減しない
```
