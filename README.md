# MovieLens 1M 映画推薦アプリ

MovieLens 1M の評価データを使い、NCF (PyTorch) で映画を推薦する試作アプリです。  
入力された評価から利用者ベクトルを作る fold-in と、重み λ によるジャンル多様性の再ランキングを行います。  
推薦は10件で、予測評価とジャンルを表示します。

## 動作確認環境

- WSL2 / Ubuntu 26.04 LTS
- CPU
- Python 3.12.11 (`python --version`)

## 環境構築

```bash
python3 -m venv dla
source dla/bin/activate
pip install -r requirements.txt
```

動作確認した版(WSL2 / Ubuntu、CPU): torch 2.14.1+cpu、numpy 2.5.4、pandas 3.1.0、streamlit 1.66.1、pytest 9.1.1。`requirements.txt` は直接使うパッケージだけを列挙していて、版は固定していません。

CPU だけの環境で、軽量版の PyTorch を入れる場合は、`pip install -r requirements.txt` の前に、次を実行してください(任意)。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## MovieLens 1M データの取得

1. [MovieLens 1M の公式ページ](https://grouplens.org/datasets/movielens/1m/)から `ml-1m.zip` をダウンロードします。
2. リポジトリのルートで、ダウンロードしたzipを `data/` に展開します。以下はzipをリポジトリのルートに置いた場合です。

   ```bash
   mkdir -p data
   unzip ml-1m.zip -d data/
   ```

3. `data/ml-1m/` に次のファイルがあることを確認します。`data_utils.py` が評価と映画情報の読込に使います。

   ```text
   data/ml-1m/ratings.dat
   data/ml-1m/movies.dat
   ```

MovieLens のデータを再配布しないでください。商用に使用せず、GroupLens がこのアプリや推薦内容を推奨していると示唆しないでください。データは repository に含めません。

引用:

> Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

## 実行手順

データを配置した後、次の順で実行します。

1. 利用者単位で train / dev / test に分割します。

   ```bash
   python make_split.py
   ```

   `splits/train_users.txt`、`splits/dev_users.txt`、`splits/test_users.txt` と、dev/test の入力・評価データ `splits/dev_inputs.json`、`splits/test_inputs.json` が作成されます。

2. モデルを学習します。`--model` は `mean`、`mf`、`ncf` から選べます。

   ```bash
   python train.py --model mean
   python train.py --model mf
   python train.py --model ncf
   ```

   それぞれ `models/mean.pt`、`models/mf.pt`、`models/ncf.pt` に保存されます。`models/` の学習済みモデルは repository に含めないため、利用前に学習が必要です。アプリはNCFを使うため、少なくとも `python train.py --model ncf` を実行してください。

3. Streamlit アプリを起動します。

   ```bash
   streamlit run app.py
   ```

## 画面の使い方

- 映画を3本以上選び、それぞれの評価を1〜5で入力します。
- ジャンル多様性の重み λ の初期値は `0.25` です。
- 推薦は10件です。並び順は「予測評価 + λ × ジャンルの新しさ」で決まります。表示される予測評価には、この加点は含まれません。
- `python check_model.py` を実行すると、サンプル入力からfold-in・予測・推薦までの値を表示して確認できます。

## 評価

dev の全モデル・設定を評価するには、次を実行します。

```bash
python evaluate.py splits/dev --all
```

test の評価は人間だけが実行し、運用上2回までにしてください。実行には `--all` が必要で、スクリプトが実行前に確認を求めます。履歴は `results/test_runs.log` に記録されます。確認には人が応答し、自動入力で通さないでください。

## 主な設定

| 項目 | 設定 |
|---|---|
| 乱数seed | 42 |
| 分割 | 利用者単位で train 80% / dev 10% / test 10% |
| 既定 λ | 0.25 |
| fold-in | l2 = 1.0、steps = 50、lr = 0.05 |

fold-in と λ の設定・決定根拠は [`docs/DEFAULT_LAMBDA.md`](docs/DEFAULT_LAMBDA.md) と [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) を参照してください。

## テスト

```bash
python -m pytest -q
```

## ファイル構成

| ファイル / ディレクトリ | 役割 |
|---|---|
| `make_split.py` | 利用者単位でデータを分割 |
| `train.py` | mean / MF / NCF モデルを学習 |
| `evaluate.py` | 指定した分割でモデルを評価 |
| `app.py` | Streamlit の推薦画面 |
| `recommender.py` | fold-in、予測、再ランキング |
| `data_utils.py` | MovieLens データの読込とID対応表の作成 |
| `models.py` | MF / NCF のモデル定義 |
| `check_model.py` | サンプル入力による推論の確認 |
| `metrics.py` | 評価指標 |
| `data/ml-1m/` | MovieLens 1M データ（repository に含めない） |
| `splits/` | 分割結果（生成物） |
| `models/` | 学習済みモデル（生成物、repository に含めない） |
| `results/` | 評価結果・test 実行履歴 |
| `docs/` | 設計・設定・失敗分析などの文書 |

## 制限

試作段階のため、入力が数本だけでは個人化が弱くなることがあります。dev では NCF の P@10 が人気順より低く、改善の余地があります。MovieLens 1M は2000〜2003年のデータで、評価では利用者が実際に評価した映画しか正解として扱えません。詳細は [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) を参照してください。
