# automatic-ml-model

画像からマルチモーダルLLM（VLM）でメタデータ（特徴量）を抽出し、LightGBM で分類/回帰モデルを作る自律開発パイプライン。

特徴量の設計（項目とVLMへのプロンプト）はローカルLLMが行い、val スコアが合格ラインを超えるまで **人間の介入なしに** 「設計 → 抽出 → 学習 → 評価 → 再設計」を繰り返す。

```
① 問題設定・アノテーション（Streamlit / 人間）
        ↓
② train/val/test split（1回だけ・以降固定）
        ↓
┌─ ③ 特徴量スキーマ設計/改訂  … ollama launch claude (qwen3.6:35b-a3b-coding-nvfp4)
│       ↓
│  ④ メタデータ抽出           … ollama chat + structured outputs (qwen3.5:9b)
│       ↓
│  ⑤ LightGBM 学習・val評価
│       ↓
└─ ⑥ val < 合格ライン なら ③ へ（診断レポートをフィードバック）
        ↓ val ≥ 合格ライン
⑦ test で最終評価 → results/final_report.md
```

## 前提

- [uv](https://docs.astral.sh/uv/)、[ollama](https://ollama.com)（`qwen3.6:35b-a3b-coding-nvfp4` と `qwen3.5:9b` をDL済み）
- macOS では LightGBM 用に `brew install libomp`

## 使い方

```bash
# 1. 問題設定 + 画像アップロード + アノテーション（ブラウザが開く）
uv run streamlit run app/annotate.py

# 2. train/val/test 分割（70/15/15）
uv run python -m pipeline.split

# 3. 自律改善ループ開始（以降は完全無人。Ctrl+C で中断→再実行で再開）
uv run python -m pipeline.run_loop
```

完了すると `results/final_report.md` に test スコア・混同行列・特徴量スキーマの変遷が出力される。

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
| `data/splits/` | train/val/test 分割（`--force` なしでは上書き不可） |
| `schemas/metadata_v{N}.json` | イテレーションNの特徴量スキーマ（項目・型・VLMプロンプト） |
| `features/features_v{N}.csv` | 抽出済みメタデータ（差分抽出のキャッシュ兼用） |
| `models/model_v{N}.txt` | LightGBM モデル |
| `results/iter{N}/report.md` | 診断レポート（designerへのフィードバック） |
| `results/final_report.md` | 最終レポート（test評価） |
| `state.json` | ループ進行状態（削除すると最初からやり直し） |

## 設計メモ

- **合格判定**: 分類 = val macro-F1、回帰 = val R²。閾値は `config.yaml` の `target_metric_threshold`（既定 0.9）。達成まで無制限にループする。
- **test リーク防止**: test は最終評価の1回しか使わない。split の再生成には `--force` が必要。
- **差分抽出**: 特徴量の再抽出要否は designer の申告（action）ではなく、定義（type/choices/prompt）の同一性で判定する。変更のない特徴量は前バージョンのCSVから流用。
- **再開性**: スキーマ・特徴量CSV（1画像ごと追記）・state.json が全てディスクに残るため、どのフェーズで中断しても再実行で続きから走る。
- **障害耐性**: designer の不正JSONはエラー内容を添えて最大3回リトライ。VLM抽出の失敗はリトライ後 NaN（LightGBM が NaN を扱える）。
