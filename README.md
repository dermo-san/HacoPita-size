# HacoPita — 箱ID推定CSVアプリ（Renderデプロイ）

## 1. 目的
HacoPita は、CSVをアップロードすると `model.pkl`（Azure AutoML 由来の学習済みモデル）で箱ID（box_id）を推定し、推定結果を追記したCSVをダウンロードできるWebアプリです。

## 2. 重要な前提（このREADMEが仕様）
このリポジトリは、以下を「絶対に」満たす実装で固定します。

- 推論は `artifacts/model.pkl` を **joblib でロード**し、`model.predict()` を呼び出す
- 推論入力に使う特徴量カラムは **固定リスト**（`app/constants.py`）のみ
  - 入力CSVに余分な列があっても、推論に混ぜない
  - 必須列が欠けていたら 400 を返す
- 出力CSVは入力の全列を保持しつつ、`box_id_pred` を追加
  - `box_id` 列が存在し空欄の行のみ `box_id_pred` で補完（既存値は上書きしない）
- 依存関係は `conda_env_v_1_0_0.yml` の指定を正とする（勝手に削減しない）
  - `model.pkl` が Azure AutoML 由来で、`azureml` 系モジュールが import されるため

## 3. リポジトリ構成

```
.
├── app/
│   ├── main.py              # FastAPI エントリ
│   ├── inference.py         # 前処理・推論・後処理
│   ├── constants.py         # 特徴量カラム固定リスト等
│   └── templates/index.html # アップロード画面
├── artifacts/
│   └── model.pkl            # 学習済みモデル（必須）
├── tests/                   # 可能な範囲で
├── requirements.txt
├── Dockerfile
└── README.md
```

## 4. API
- `GET /` : アップロードフォーム
- `POST /predict` : CSVを受け取り推論し、推論済みCSVを返す（attachment）
- `GET /healthz` : 200 OK を返す

## 5. 入力CSV仕様
- 入力CSVは列数が多くてよいが、推論に使うのは `FEATURE_COLUMNS` のみ
- 必須列が欠けている場合は 400（欠損列名を列挙）
- 数値化は `pd.to_numeric(errors="coerce")` → 欠損は 0 補完（推論が落ちないことを優先）
- 文字コードは `utf-8` を優先し、失敗したら `cp932` を試す

## 6. 出力CSV仕様
- `box_id_pred` を新規追加
- `box_id` 列が存在し空欄の行のみ `box_id_pred` で補完
- 返却CSVは `utf-8-sig`（Excel互換を優先）
- 返却ファイル名例：`{元ファイル名}_with_predictions.csv`

## 7. ローカル実行（Docker推奨）

### 7.1 ビルド

```bash
docker build -t hacopita .
```

### 7.2 起動

```bash
docker run --rm -p 8000:8000 -e PORT=8000 hacopita
```

ブラウザで http://localhost:8000/ を開き、CSVをアップロードしてください。

## 8. Renderデプロイ
- Render の Web Service としてこのリポジトリを接続
- Docker を使う（推奨）
- 起動は以下が前提：
  - `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## 9. 開発上の注意（破壊的変更を禁止）

以下は不具合や再現性崩壊の原因になるため禁止します。
- 特徴量カラム名・順序の変更
- 推論方式（predict / predict_proba）の勝手な変更
- 依存ライブラリの“軽量化”目的の削除
- 入出力CSVの列の勝手な削除

## 10. トラブルシュート

### 10.1 ModuleNotFoundError: No module named 'azureml'

model.pkl が Azure AutoML 由来のため、azureml 依存が必要です。
requirements / Dockerfile の依存を削っていないか確認してください。

### 10.2 必須列がない（400）

入力CSVに必要列が欠けています。レスポンスに出る欠損列を追加するか、
学習時と同じ特徴量生成処理を入力側で行ってください。

---

### 依存関係の扱いについて（重要な補足）
`model.pkl` をローカルでロードするには、Azure AutoML 由来の内部クラス参照のため **azureml系の依存関係が必要**になりやすいです。添付の conda 環境定義（Python 3.10.19、`azureml-train-automl-runtime==1.61.0`、`azureml-defaults==1.61.0`、`scikit-learn==1.5.1` など）をベースに固定してください。 [oai_citation:2‡conda_env_v_1_0_0.yml](sediment://file_0000000020587206b5022656554e2c66)
