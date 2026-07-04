# 画像収集: 収集計画の設計と実行

## 収集プランJSON（`data/collection_plan.json`）

`scripts/collect_images.py --plan` に渡す形式:

```json
{
  "categories": {
    "bucket_name": {
      "queries": ["english search query", "..."],
      "keywords": ["relevance", "filter", "words"],
      "synthetic": null,
      "per_category": 50
    }
  },
  "base_pool_queries": ["..."],
  "base_pool_keywords": ["..."]
}
```

- `queries`: 必須・非空。**英語**で書く（画像APIの検索は英語が圧倒的に強い）。1バケットに3〜6本、言い回しを変えて用意する
- `keywords`: title/tags に対する関連性フィルタ（小文字比較・部分一致）。省略/空 = フィルタなし。**汎用名詞を1〜3語**（例: `["dog", "puppy"]`）。厳しすぎると収集がほぼゼロになる
- `synthetic`: `motion_blur` / `backlit` / `cropped_face` / `unclear_face` または null。指定時は `base_pool_queries` 必須
- `per_category`: バケット毎の目標枚数。省略時は `--per-category`（デフォルト50）
- バケット名: 英小文字始まりの `[a-z0-9_]+`。`_base_pool` は予約名。ファイル名の接頭辞になる

## バケット設計の指針

**バケット = 検索の網であり、ラベルではない**。設計原則:

1. **各クラスに最低1バケット**。そのクラスの画像が集まりそうなクエリ群を持たせる
2. **紛らわしい負例のバケットを足す**。境界を学習させるため（例: 「犬あり/なし」なら `cat_photo` や `empty_park` バケットで「犬に似た/犬がいそうで居ない」画像を確保）
3. **1バケットに詰め込みすぎない**。クエリの方向性が違うなら分ける（診断しやすくなる）
4. クラス分布はアノテーション後に決まるので、バケット枚数はクラス目標の1.2〜1.5倍を見込む（DISCARD・誤バケット混入のバッファ）

## synthetic 加工の使いどころ

デフォルトは **null**。検索でほぼ集められない視覚状態（ブレ・逆光・見切れ・ボケ）のクラスがあるときだけ使う。

注意: 加工画像は「加工アーティファクト自体」がラベルと相関するリークを生む。加工で作ったクラスとそうでないクラスを分類するモデルは、本物の分布で性能が出ない可能性がある。使う場合は最終報告に明記する。

## ソースの選択

| ソース | キー | 備考 |
|---|---|---|
| openverse | 不要 | デフォルト。`category=photograph` 指定済み（実装内） |
| wikimedia | 不要 | デフォルト |
| pixabay | `PIXABAY_API_KEY` | .env にあれば auto で使われる |
| pexels | `PEXELS_API_KEY` | 同上 |
| flickr | `FLICKR_API_KEY` | **新規APIキーは有料Proのみ**。既存キーがある場合だけ |
| google | `GOOGLE_API_KEY`+`GOOGLE_CSE_ID` | license=unknown になるため **デフォルト除外を維持**。明示指定時のみ |

`--sources auto`（デフォルト）で .env のキー有無に応じて自動選択される。キーレス環境では openverse + wikimedia だけで動く。

## 既知の落とし穴

- **クリップアート/イラスト混入**: openverse は photograph 指定済みだが他ソースは混じる。Phase 2 のスポットチェックで目視確認し、混入バケットはクエリに `photo` を足すか keywords で絞る。アノテーション時の DISCARD でも最終防衛できる
- **keywords が厳しすぎて0枚**: keywords は「無関係画像の除外」用であり「正例の証明」用ではない。収集が目標の70%未満なら真っ先に keywords を緩める
- **レート制限**: openverse は匿名3秒/リクエスト。バケット×クエリが多いと時間がかかる（50枚×4バケットで30分〜1時間程度を見込む）。バックグラウンド実行してポーリングする
- **再実行は安全**: `collection_metadata.csv` を読んで取得済みをスキップするので、クエリを追記して同じコマンドを再実行すれば差分収集になる

## クエリ拡張ループ（目標未達時）

1. `collection_metadata.csv` でバケット毎の到達数を確認
2. 70%未満のバケット: (a) keywords を減らす/空にする → (b) クエリの言い換え・上位概念・関連シーンを3本追加 → (c) `--sources` にキー付きソースを足せないか確認
3. プランJSONを編集して同じコマンドを再実行（差分収集）
4. **最大2ラウンド**。それでも未達なら現有枚数で先へ進み、ユーザーに不足を報告（クラス枚数のハードフロアは annotation.md 参照）

## ライセンス

`collection_metadata.csv` に license / creator が全画像分残る。openverse/wikimedia/flickr は CC 系（**NC = 非商用限定を含む**）。ユーザーの用途が商用なら、最終報告でNC/ND混入の有無と件数を必ず報告する。
