# Copilotへの依頼プロンプト一覧(映画推薦アプリ)
手順の全体は `IMPLEMENTATION_MANUAL.md` を参照。各promptは**全文をそのまま**貼り付ける。
共通の前提(ルール、評価の定義)は repository 直下の `CONTEXT.md` にあり、各promptの1行目で読ませる。
使い方：毎回「変更計画を示させる → 自分で読む → `計画どおりに実装して` → 自分で実行して確認 → 自分でcommit」の順にする。

## 不具合が出たときの依頼の型
```text
【期待】…になるはず
【実際】…と表示された(コマンドと出力を貼る)
【疑い】前処理 / モデル / 後処理 のどこか(根拠)
まず原因の候補と、確かめる方法を示して。修正は私が確認してから。
```

---

## S4 data_utils.py

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画(関数名・引数・戻り値)を先に示し、私が承認するまで実装しない。
data_utils.py を作る。
- load_ratings(path): data/ml-1m/ratings.dat を読み、列 user_id, movie_id, rating, timestamp の DataFrame を返す。区切りは "::"
- load_movies(path): data/ml-1m/movies.dat を読み、列 movie_id, title, genres(list[str]) の DataFrame を返す。ジャンルは "|" 区切り。文字コードは実際のfileに合わせ、どう確かめたかも示す
- build_id_maps(ratings_train): train の評価に現れる movie_id だけに連番index(0始まり)を振り、movie_id→index と index→movie_id の dict を返す。user_id も同様
- 全関数に docstring(戻り値の列名と型)を書く
- python data_utils.py で、ratings の行数、movies の行数、先頭5行を表示する __main__ を付ける
```

---

## S5 make_split.py

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
make_split.py を作る。data_utils.py の関数を使う。
- 利用者ID(6040人)を seed=42 で並べ替え、前から train 80%(4832人)、dev 10%(604人)、残り test 10%(604人)に分ける
- splits/train_users.txt, dev_users.txt, test_users.txt に利用者IDを1行1個で保存する
- 3集合の共通部分が空であることを assert で確認する
- dev と test の各利用者について、その利用者の評価から5本を seed固定でランダムに選んで「入力用」とし、残りを「評価用」とする
- splits/dev_inputs.json と splits/test_inputs.json に保存する。形式は {"<user_id>": {"input": [[movie_id, rating], ...5件], "eval": [[movie_id, rating], ...]}}
- 実行すると、各集合の人数と評価件数だけを表示する。testの中身(評価値の統計など)は表示しない
- 同じコマンドを2回実行しても、同じfileができること
```

---

## S6 metrics.py とテスト

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
metrics.py と tests/test_metrics.py を作る。
metrics.py:
- rmse_pooled(preds, trues): 全評価をまとめた RMSE
- rmse_macro(list_of_(preds, trues)): 利用者ごとの RMSE の平均
- high_threshold(eval_ratings, mode): mode="top25" なら numpy.percentile(eval_ratings, 75, method="lower") を返す。評価が全部同じ値なら None を返す。mode="fixed4" なら 4 を返す
- precision_at_k(recommended_ids, relevant_ids, k=10): 上位k件のうち relevant に含まれる割合
- genre_count(list_of_genre_lists): 重複なしのジャンルの種類数
tests/test_metrics.py (私が手計算した次の値で検証する。実装のcodeから期待値を作らない):
- rmse_pooled([3.5,3,4],[4,3,5]) は約 0.6455 (小数4桁で一致)
- high_threshold([5,5,4,4,4,4,3,3,2,1],"top25") は 4。このとき評価が4以上の件数は 6
- high_threshold([3,3,3],"top25") は None
- high_threshold(任意,"fixed4") は 4
- precision_at_k: 上位10件のうち3件が relevant なら 0.3
- genre_count([["Action","Comedy"],["Action"],["Drama"]]) は 3
```

---

## S7 train.py(平均・行列分解)

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
models.py と train.py を作る(この段階ではNCFは作らない)。
- models.py: MF (行列分解)。予測 = 全体平均 + 利用者偏り + 映画偏り + 利用者ベクトルと映画ベクトルの内積。PyTorchのnn.Module。fold-inで利用者ベクトルと利用者偏りを外から与えられる関数 predict_with_user_params(user_vec, user_bias, item_idx) も用意する
- train.py: 引数 --model {mean,mf} --dim(既定32) --epochs(既定20) --lr(既定1e-3) --wd(既定1e-5) --batch(既定1024) --seed(既定42) --shrink(mean用、既定10)
- data/ml-1m と splits/train_users.txt から、train利用者の評価だけを読む(dev/testの利用者は読まない)
- train利用者の評価のうち5%を検証用(val)に取り分け、学習には使わない。early stoppingはvalのRMSEで行う(数epoch悪化したら停止)
- --model mean: trainでの映画ごとの平均評価を、(合計 + shrink×全体平均)/(件数 + shrink) で計算して保存する
- --model mf: MSE損失、Adam。各epochの train損失と val RMSE を表示する
- 保存するもの: models/ に、重み、ID対応表、全体平均、train利用者ベクトルの平均(fold-inの初期値用)、映画ごとのtrain評価件数。保存先のfile名を一覧で示す
- CPUで1回の学習が数分〜十数分で終わるようにする
```

---

## S8 NCFの追加

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
models.py と train.py に NCF を追加する(既存のMFとmeanの動作は変えない)。
- NCF: 利用者埋め込み(dim)と映画埋め込み(dim)を連結(2*dim)→ 64 → 32 → 16 → 1 のMLP(中間層はReLU、出力は活性化関数なし)。予測 = 全体平均 + MLPの出力。Dropoutは引数 --dropout(既定0.1)
- fold-in用に、利用者ベクトルを外から与えて予測する関数 predict_with_user_params(user_vec, item_idx) を用意する
- train.py に --model ncf を追加。損失はMSE、Adam、valのRMSEでearly stopping
- 保存するもの: MFと同じ形式(重み、ID対応表、全体平均、train利用者ベクトルの平均など)
- 各epochの train損失と val RMSE を表示する
```

---

## S9 recommender.py

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画(関数名・引数・戻り値)を先に示し、私が承認するまで実装しない。
recommender.py と tests/test_recommender.py を作る。画面(Streamlit)には依存させない。
recommender.py:
- load_model(name): models/ から読み込む。name は "mf" または "ncf"。呼び出し側が1回だけ呼べる形にする
- validate_inputs(items): items は [(movie_id, rating), ...]。3本未満、同じ映画の重複、データにない映画、評価が1〜5の整数でない場合は ValueError(理由を日本語で)
- to_movie_index(items, model): movie_id を連番indexに変換する
- fold_in_user(model, item_idx, ratings, steps=50, lr=0.05, l2=0.01): モデルの重みを固定し、利用者のパラメータだけをAdamで最適化して返す。初期値は train利用者ベクトルの平均(偏りは0)。損失 = 入力評価のMSE + l2 × (初期値からの距離の2乗)。最初と最後の損失も返す
- predict_ratings(model, user_params): 全映画の予測評価を1〜5にclipして返す
- rerank(candidates, lam, k=10): candidates は [(movie_id, pred, genres), ...]。上位から1本ずつ選ぶ。選択済みの映画のジャンルに含まれないジャンルの割合(0〜1) × lam を予測評価に加えた値が最大のものを次に選ぶ。同点は movie_id 昇順。lam=0 なら予測評価順になる
- recommend(items, lam=0.25, model_name="ncf", debug=False): 上記をつなぎ、見た映画を除いて [(title, pred, genres), ...] を10件返す。debug=True のとき、モデルへ渡す直前のindex配列のshape・dtype、評価のmin/max、fold-inの最初と最後の損失、生の予測の上位5件を表示する
tests/test_recommender.py (私が手計算した次の例で検証する):
- validate_inputs: 2本だけ、同じ映画2回、評価6、評価0 はいずれも ValueError
- rerank: 候補 A(pred 5.0, [Action])、B(4.9, [Action])、C(4.0, [Comedy])、k=2。lam=0 なら A,B の順。lam=1 なら A,C の順
```

---

## S9-fallback fold-inの代替

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
recommender.py に、fold-in の代替方法を追加する(既存の関数は変えない)。
- fold_in_embedding_average(model, item_idx, ratings): 入力映画の学習済み映画ベクトルを、評価(1〜5)から全体平均を引いた値で重みづけして平均し、利用者ベクトルとする。これをNCFまたはMFの利用者ベクトルとして使い、predict_ratings で全映画を予測できるようにする
- recommend() に引数 fold_in_method ("optimize" または "average"、既定 "optimize") を追加する
- tests/test_recommender.py に、average方式でも ValueError の検証と rerank が変わらないことを確認するテストを追加する
```

---

## S10 evaluate.py

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
evaluate.py を作る。metrics.py と recommender.py の関数を使う。
- 使い方: python evaluate.py splits/dev --model ncf --lam 0.25 [--limit N]
- 引数で指定した分割(splits/dev または splits/test)のfileだけを読む。ディレクトリ全体を再帰的に読まない
- --model は mean / popular / random / mf / ncf。mean は train の映画平均(縮小あり)で順位づけ、popular は train の評価件数順、random は seed固定の無作為。mf と ncf は各利用者の入力5本でfold-inしてから予測する
- 候補は入力以外の全映画。trainに無い映画は候補と評価から除き、その件数を表示する
- 指標: RMSE(pooled を主、macro を副)、Precision@10(主: 利用者ごとの上位25%以上 = high_threshold(..., "top25")、副: 4点以上)、ジャンル数(上位10件)。すべて metrics.py の関数を使う
- 主指標で除外した利用者数(評価が全部同じ値)と、利用者ごとの高評価の割合の平均・中央値も表示する
- 結果を表で表示し、results/ にcsvで保存する(ファイル名に分割名・モデル・lam・日時を入れる)
- 分割が test のときは、実行前に「testの評価は2回までです。続けますか? (yes/no)」と確認し、yes のときだけ実行して results/test_runs.log に日時・モデル・lamを追記する。no なら何もせず終了する
- --limit N のときは先頭N人だけ評価する(動作確認用)
- --all: mean / popular / random / mf と、ncf の全λ(0, 0.1, 0.25, 0.5, 1.0)を**1回の実行**でまとめて評価し、結果を表とcsvに保存する。testの確認と実行履歴は、この1回の実行を1回として数える
- 分割が test のときは --all のみ許可する。--model を個別に指定した場合は「testは --all でだけ実行できます」と表示して終了する
```

---

## S11 check_model.py

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
check_model.py を作る。recommender.py の関数を使う。
sample入力(SF映画の高評価5本。movies.dat から実在する5本のmovie_idを選び、タイトルと一緒に表示する。評価は5,5,5,4,4)に対して、次を表示する:
- 入力した(title, rating)
- 変換後のindex配列のdtypeとshape、評価のmin/max
- fold-inの最初と最後の損失
- モデルの生の出力(clip前)の上位5件
- recommend() の最終結果の上位5件(映画名・予測評価・ジャンル)
- fold-in + 全映画の予測にかかった時間(time.perf_counter)と、実行環境(CPU/CUDA)
```

---

## S12 app.py

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
app.py を作る(すでに app.py がある場合は、先に既存の内容を読み、何を残し何を置き換えるかを変更計画に書く)。recommender.py の recommend() と validate_inputs() を呼ぶだけにする。
- 映画は検索つきの選択欄(st.multiselect または st.selectbox)で選び、評価を1〜5で入力する。データにある映画だけ選べる
- 3本以上で「おすすめを表示」ボタンを押せる
- 不正な入力は st.error で理由を日本語で表示し、推薦しない
- モデルの読込は @st.cache_resource で1回だけにする
- λ は st.slider(0.0〜1.0、刻み0.05、初期値0.25)
- 結果は映画名・予測評価・ジャンルの表(順位つき)で表示する
- 画面構成はこれ以上増やさない
```

---

## S13 README.md

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
README.md を作る(既存のREADME.mdがあれば、その内容を残して拡張する)。
- アプリの説明(2〜3行)
- 環境構築: python3 -m venv dla / source dla/bin/activate / pip install -r requirements.txt
- データの取得: MovieLens 1M の公式URLからダウンロードして data/ml-1m/ に展開する手順。再配布しないこと、出典(GroupLens)、licenseは公式READMEを確認すること
- 実行手順(順番どおり): python make_split.py / python train.py --model mf ... / python train.py --model ncf ... / streamlit run app.py
- 評価の実行方法: python evaluate.py splits/dev --model ncf --lam 0.25 。testは2回までと明記
- 現在の制限(試作段階)
```
