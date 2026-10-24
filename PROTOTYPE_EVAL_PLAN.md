# 第8回 試作品と評価計画

- 学籍番号：2432059
- 氏名：平田琉翔
- 提出日：2026-10-24（最終commit時に確認）
- 提出file名：`PROTOTYPE_EVAL_PLAN.md`（作成するアプリのrepository直下に置き、commitして提出する）
- 締切：**10/25（日）23:59**（次の講義が11/5のため、例外として休講前に提出する）

> 実測値（3章・4章・6章）は、実際に実行した結果である。認証情報、password、API keyは記録していない。

---

## 1. 企画からの変更点

| 項目 | 企画（`IDEA_PROPOSAL.md`） | 現在 | 変更した理由 |
|---|---|---|---|
| 比較対象 | 平均評価と行列分解 | 平均評価・行列分解に加え、**人気順**（評価数が多い順）と無作為を追加（採用） | 平均評価だけでは評価数の少ない映画が上位に出やすく、Precision@10の基準として弱い。「簡単な方法」と比べるため。 |
| testの使い方 | λの全水準をtestで報告 | 同じ。ただしtestの評価は**2回だけ**（改善前・改善後）とし、既定のλはdevだけで決める | 第8回講義の「testは最後に使う」に合わせて明確にした |
| 入力本数 | 「数本」 | 評価用ユーザーの入力は**5本に固定** | 評価の再現性のため。画面では3本以上を受け付ける |

（上記以外の変更：なし）

---

## 2. 処理の流れ

```text
入力 → 前処理 → [モデル] → 後処理 → 表示
```

| 段階 | 処理の内容 | 深層学習 / ルール | 担当するfile・関数 |
|---|---|---|---|
| 入力 | データセット内の映画を選び（検索つき選択欄）、評価1〜5を入力。3本未満・重複・範囲外は受け付けない | ルール | `app.py`、`validate_inputs()` |
| 前処理 | 映画名 → movieId → 埋め込み用の連番indexへ変換。評価を数値にそろえる | ルール | `recommender.py`、`to_movie_index()` |
| モデル | 学習済みNCFの映画埋め込みを固定し、入力評価に合うよう**利用者ベクトルだけ**を最適化（fold-in）。そのベクトルで未視聴の全映画の評価を予測 | **DL** | `recommender.py`、`fold_in_user()`、`predict_ratings()` |
| 後処理 | 見た映画の除外 → ジャンル多様性の再ランキング（重みλ。順位づけはclip前の予測値で行う）→ 上位10件 | ルール | `recommender.py`、`rerank()` |
| 表示 | 映画名・予測評価（1〜5にclip、小数2桁）・ジャンルを表で表示 | ルール | `app.py` |
| （事前の学習） | MovieLens 1MのtrainユーザーでNCF・行列分解を学習し、重みを保存 | **DL** | `train.py`（画面とは別に1回実行） |

> `app.py` は `recommender.py` の関数を呼ぶだけにし、評価script `evaluate.py` も同じ関数を使う（講義p.11〜12）。
> Streamlitは操作のたびに再実行されるため、モデルの読込は `@st.cache_resource` で1回にする。

---

## 3. モデルの仕様

- モデル名：Neural Collaborative Filtering（NCF）。PyTorchで自作（比較対象：行列分解）
- 入手元（URL）：自作（学習データ：MovieLens 1M、https://grouplens.org/datasets/movielens/1m/ ）
- license：MovieLens 1Mの公式README（`data/ml-1m/README`）による条件：研究目的での利用可／GroupLens・ミネソタ大学の推奨を示唆しない／成果物でのデータ利用の明記と引用（Harper & Konstan, 2015, ACM TiiS 5(4), Article 19, DOI 10.1145/2827872）／**許可なく再配布しない**／許可なく商用・収益目的で使わない。本課題は授業の非営利の利用である。データと、評価データを含むfile（`splits/*_inputs.json`）はrepositoryに含めず、READMEに取得手順を書く。PyTorch 2.14.1（CPU版）はpipのメタデータで Apache-2.0、BSD-2-Clause、BSD-3-Clause、BSL-1.0、MIT などの複合表記（詳細は公式のLICENSEで確認）
- 入力（形式・shape・値域）：利用者ベクトル `(d,)` 〔d=32（`train.py` の `--dim` の既定値）〕、映画index（整数）。評価は1〜5の整数
- 出力（形式・意味）：予測評価（実数）。順位づけにはclip前の値、画面の表示とRMSEの計算には1〜5にclipした値を使う
- 実行環境（CPU / CUDA / MPS）と1回の推論時間：CPU（WSL2上、PyTorch 2.14.1+cpu）。fold-in＋全映画の予測：初回 2.57秒、ウォームアップ後 0.147秒（`time.perf_counter()`で計測）

モデルへ渡す直前と、モデルの出力を表示した結果：

```text
$ python check_model.py   （モデル: ncf / 実行デバイス: CPU）
- 入力：[(movie_title, rating), ...] 5本
  (Star Wars: Episode IV - A New Hope (1977), 5)
  (Star Wars: Episode V - The Empire Strikes Back (1980), 5)
  (Star Wars: Episode VI - Return of the Jedi (1983), 5)
  (Matrix, The (1999), 4)
  (Terminator 2: Judgment Day (1991), 4)
- 変換後：movie index の配列 dtype=int64, shape=(5,)；評価 min=4.0, max=5.0
- fold-in：初期の利用者ベクトルはtrain利用者ベクトルの平均、shape (32,)。最適化はAdam、step数 50、lr 0.05、初期値からの二乗距離のl2係数 0.01。
  損失：最初=0.349100、最後=0.185436
- モデルの生の出力：クリップ前の予測評価の上位5件（movie_id, 映画名, 予測, ジャンル）
  (318, Shawshank Redemption, The (1994), 4.9761, ['Drama'])
  (1198, Raiders of the Lost Ark (1981), 4.8633, ['Action', 'Adventure'])
  (260, Star Wars: Episode IV - A New Hope (1977), 4.8627, ['Action', 'Adventure', 'Fantasy', 'Sci-Fi'])   ← 入力した映画も含む
  (50, Usual Suspects, The (1995), 4.8603, ['Crime', 'Thriller'])
  (2905, Sanjuro (1962), 4.8414, ['Action', 'Adventure'])
- 画面に表示した値：recommend() の上位5件（入力済みの映画を除外。λ=0.25は再ランキング後。表示の予測は加点を含まない）
  λ=0   ：Shawshank Redemption 4.9761 / Raiders of the Lost Ark 4.8633 / Usual Suspects 4.8603 / Sanjuro 4.8414 / Godfather, The 4.8389
  λ=0.25：Shawshank Redemption 4.9761 / Raiders of the Lost Ark 4.8633 / Usual Suspects 4.8603 / Close Shave, A 4.8368 / Casablanca 4.7870
  （λ=0.25では4位・5位が、すでに選ばれたジャンルと重ならない映画に入れ替わった）
```

---

## 4. 動作確認（入力から出力まで）

```text
$ git log --oneline
49e36b9 Add dev results after ranking fix and dev log note
69e2069 Rank by unclipped predictions; clip only for display and RMSE
f5ee139 Add Streamlit app for movie recommendation
c67c79e Add check_model.py for input/output verification
795a8fe Filter unmapped input movies in evaluation and add dev results
9ae7e80 Filter unmapped input movies in evaluation and add dev results
95b81c9 Add evaluation script
38c4fa0 Add recommender with fold-in and diversity reranking
6926e05 Add training results note
f95c8bb Add NCF training
659368d Add mean and matrix factorization training
92783c3 Add evaluation metrics with hand-computed tests
8291de8 Add dev log note for split step
2f94512 Add user-level train/dev/test split
8f8f58d Add data loading and id map utilities
b85ef22 Add requirements
0741754 Add Copilot context and dev log notes
46c4dba Add proposal, evaluation plan, design docs
3404cfe Add gitignore
3923580 Fill in student info
c8638e8 Add IDEA_PROPOSAL.md
（この後、本ファイルのcommitが加わる）

$ python -m pytest -q
23 passed

$ streamlit run app.py
ブラウザで http://localhost:8501 に画面が開き、入力・推薦の表示まで動いた（下の表）。
```

| 入力 | 事前の期待 | 画面に表示された結果 | 期待どおりか |
|---|---|---|---|
| 同じジャンル（例：SF）の高評価5本 | SF系が多く上位に出る（λ=0） | 入力：Star Wars IV/V/VI＝5、Matrix＝4、Terminator 2＝4、λ=0。上位10件（予測）：Shawshank Redemption（4.98）、Raiders of the Lost Ark（4.86）、Usual Suspects（4.86）、Sanjuro（4.84）、Godfather（4.84）、Close Shave, A（4.84）、Schindler's List（4.82）、Great Escape, The（4.82）、It's a Wonderful Life（4.81）、Casablanca（4.79）。ジャンル数9。Sci-Fiは0本 | **×（期待と違った）** |
| 異なるジャンル5本（高評価・低評価が混在） | 低評価の映画と近い作品は下がる | 入力：Star Wars IV＝4、Wallace & Gromit: The Best of Aardman Animation＝5、Jumanji＝4、Love Jones＝2、Human Traffic＝1、λ=0。上位10件：Godfather（4.71）、Close Shave, A（4.64）、Shawshank Redemption（4.62）、Seven Samurai（4.53）、Raiders of the Lost Ark（4.53）、Usual Suspects（4.52）、Wrong Trousers, The（4.52）、Inherit the Wind（4.51）、High Noon（4.50）、Princess Bride, The（4.49）。ジャンル数9 | △（一部のみ。低評価の映画に近い作品が下がったかは、この結果だけでは判断できない） |
| **対象外：2本だけ入力** | エラーを表示し、推薦しない | Toy Story、Jumanji の2本を選ぶと、画面に「3本以上選んでください（現在2本）」と表示され、「おすすめを表示」ボタンが押せない（無効） | ○ |
| **対象外：データにない映画名／評価6・0** | 入力を受け付けない（または拒否の表示） | 映画は選択式で、選択肢は学習データの映画だけなので、データにない映画名は入力できない。評価欄に「56」を入力すると、入力欄が赤枠になり「Press Enter to apply」と表示された（範囲外の値として扱われている）。Enter後の動作と「0」の入力は未確認。`validate_inputs()` は対応表にない映画IDを `ValueError` で拒否する（全dev評価の実行中に実際に発生して確認） | ○（Enter後の動作は未確認） |
| **想定外：5本すべて同じ評価（すべて5点）** | 嗜好の区別がなく、推薦が人気作に偏る可能性。fold-inの退化を疑う | 入力：Toy Story、Jumanji、Toys、Waiting to Exhale、Father of the Bride Part II をすべて5、λ=0.25。上位10件：Shawshank Redemption、Raiders of the Lost Ark、Usual Suspects、Star Wars IV、Schindler's List、Close Shave, A、Maltese Falcon, The、My Fair Lady、For All Mankind、Tombstone。**全件の予測評価が5.00と表示された**。入力（ファミリー・コメディ中心）と関連の薄い、評価の高い名作が並んだ | ○（人気作に偏った。予測がclipで頭打ち） |

> 少なくとも1件は対象外・想定外を試す（課題の要求）。期待と違った場合は、どの段階（前処理 / モデル / 後処理）が原因かを1つ選んで根拠を書く。

### 期待と違った結果の分析

**1行目（SFの入力で、SFが1本も出なかった）**
- 事実：SF 5本の入力でも、λ=0の上位10件にSci-Fiは入らなかった。`check_model.py` の上位5件（λ=0）でも同じ（Drama・Action・Crimeが中心）。
- 原因の段階：**モデル**（1つに絞る）。根拠：(1) NCFはジャンルを入力に使わず、評価のパターンだけから学ぶ。λはジャンルを使う唯一の箇所だが、「別のジャンルを混ぜる」向きに働くもので、同じジャンルに寄せる機能ではない。(2) 入力が5本だけのfold-inでは、個人の好みより、評価の高い人気作の傾向が強く出ると考えられる（推測）。dev全体でも人気順（P@10 0.296）がNCF（0.178）より高く、整合する。
- 未検証：同じ入力でSciFi映画の予測が実際にどの順位にあるかは、まだ調べていない。

**5行目（すべて5点で、予測がすべて5.00になった）**
- 事実：画面の予測評価は、1〜5にclipした値のため、全件が5.00になった。別の入力（アニメ5本）で生の予測を調べたところ、5.0以上の映画が67本あり、最大は5.54だった。
- 原因の段階：**モデル（fold-in）と後処理**。fold-inが高評価ばかりの入力で予測を高く寄せ、clipで頭打ちになった（推測）。さらに、修正前は後処理がclip後の値で順位づけしており、同点がmovie_id順に並んでいた。これは実際の不具合で、**修正済み**（7章）。修正後は生の値で順位づけされる。
- 残る課題：画面の表示は5.00のままで、順位の根拠が見えない。fold-inの正則化（l2・step数）を強めて予測の偏りを抑えられるかを、devで試す（8章、10/30）。効くかどうかは未検証。


---

## 5. 評価計画

### 目的

ジャンルの多様性の重み λ を変えたとき、おすすめの当たりやすさ（Precision@10）とジャンルの種類数がどう変わるかを調べる。あわせて、深層学習（NCF）が単純な方法（平均評価・人気順・無作為）より役に立つかを確かめる。

### 評価データ

MovieLens 1M（6,040人、約100万件の評価）を**利用者単位**で分ける。

| 区分 | 件数 | 集め方・条件 | 置き場所 |
|---|---:|---|---|
| train（学習用） | 利用者の80%（約4,832人） | 乱数seed固定で分割。NCF・行列分解の学習だけに使う | `data/ml-1m/`（repositoryに含めない）、分割は `splits/train_users.txt` |
| dev（調整用） | 利用者の10%（約604人） | 各利用者の評価から**5本を入力用**にランダムに選び（seed固定）、**残りを評価用**にする | `splits/dev_users.txt` |
| test（最終確認用） | 別の利用者の10%（約604人） | devと同じ方法。**分割は10/24に1回だけ作って固定**する（seed固定。利用者IDの一覧 `splits/*_users.txt` はcommitし、評価値を含む `splits/*_inputs.json` は再配布を避けるためcommitせず、seed固定で再生成する） | `splits/test_users.txt` |

- devとtestが重ならないようにした方法：利用者IDで分割し、分割script内で `assert` により3集合の共通部分が空であることを確認する。映画は3集合に共通して現れるが、映画埋め込みの学習にはtrain利用者の評価しか使わない
- testを最終評価まで使わない方法：学習率・fold-inの設定・λの既定値は**devだけ**で決める(学習の停止時期はtrain内の検証用5%で決め、devは使わない)。`evaluate.py` は評価するfileを引数で受け取り（`python evaluate.py splits/dev`）、testは評価回数を**2回（10/28：改善前、11/1：改善後）**に限る。testの結果を見て設定を変えない。AIがtestで最良の設定を選ぶcodeを提案しても採用しない

### 指標と比較対象

- 指標：
  - **RMSE**：評価用の評価（5本の入力を除いた残り）に対する予測誤差
  - **Precision@10**：候補＝入力以外の全映画。上位10件のうち、高評価だった映画の割合。高評価の基準は、**主：利用者ごとの評価用データの上位25%以上**（閾値＝75パーセンタイル(`method="lower"`)、同点を含む。全評価が同じ値の利用者は除外して人数を報告）、**副：4点以上**（結論が基準によらないかの確認）。定義は結果を見る前に固定する
  - **ジャンル多様性**：上位10件に含まれるジャンルの種類数
- 比較対象（正解・baseline）：
  - RMSE：正解は実際の評価。比較は**映画ごとの平均評価**と**行列分解**
  - Precision@10：**無作為**（下限）、**人気順**、行列分解、NCF（λ=0）
  - 再ランキングの効果：λ=0（変更前）との差
- 注意（限界として報告書に書く）：MovieLensは利用者が評価した映画しか正解がない。評価していないだけで好きな映画は「不正解」になるため、Precision@10の**絶対値は低く出る**。手法間の相対比較として読む
- 注意（限界として報告書に書く）：testは同じデータセットの別の利用者であり、別の日・別の対象で集めたデータではない。分布が近いため、未知の環境での性能までは示せない

### 比較する条件

| 条件 | 水準 | 変えないもの |
|---|---|---|
| 多様性の重み λ | 0（再ランキングなし）／0.1／0.25／0.5／1.0 | 学習済みモデル、入力本数（5本）、fold-inのstep数・lr、乱数seed、候補集合、推薦件数（10件） |

- λの尺度（予測評価に加える項の大きさ）はdevで確認し、水準は必要ならdevで調整してよい。調整後の水準をtestに適用する。
- 「どのλを既定にするか」はdevの結果だけで決める。

### 評価の実行方法

```text
$ python evaluate.py splits/dev --model mean
$ python evaluate.py splits/dev --model mf
$ python evaluate.py splits/dev --model ncf --lam 0.25
...（devは何度実行してよい）

# testは2回だけ。実行した日時と対象のcommitを下に残す
$ python evaluate.py splits/test --all     # 10/28（改善前）（未実行。実行したら日時とcommitをDEV_LOG.mdに記録する）
$ python evaluate.py splits/test --all     # 11/1（改善後）（未実行。同上）
```

---

## 6. 初期結果（devで可）

```text
dev（利用者604人。評価対象604人、全モデル同じ利用者集合）。results/dev_all_rankfix.txt
model        RMSE(pooled)  P@10(top25)  P@10(固定4点)  genres
mean         0.989         0.204        0.228          11.07
popular      N/A           0.296        0.348           9.14
random       N/A           0.012        0.015          12.00
mf           1.028         0.213        0.245           9.07
ncf(λ=0)     1.061         0.178        0.197           9.26
ncf(λ=0.1)   1.061         0.172        0.193          11.28
ncf(λ=0.25)  1.061         0.153        0.170          13.79
ncf(λ=0.5)   1.061         0.135        0.149          14.92
ncf(λ=1.0)   1.061         0.119        0.132          15.07
（高評価の割合：top25基準で平均0.479、中央値0.455。固定4点基準で平均0.624。top25除外の利用者0人。
 入力から除いた映画は延べ2件（2人）、入力3本未満で対象外にした利用者0人。
 trainに無い映画の評価13件は評価から除外。）
```

- 気づいたこと：
  - **λの比較（事実）**：λを上げると、P@10は5水準で単調に下がり（λ=0の0.178→λ=1.0の0.119）、ジャンル数は増える（9.3→15.1）。λ=0.5から1.0はジャンル数がほぼ増えず（+0.15）、P@10だけが下がる。RMSEはλによらず同じ（再ランキングは予測値を変えない設計どおり）。
  - **既定λ（暫定）**：「λ=0に対するP@10の低下が20%以内で、ジャンル数が最大のλ」を規則とすると、**λ=0.25**（−14.0%）。許容幅の20%はdevの結果を見てから決めたもので、事前に決めた規則ではない。testを見る前に固定する。
  - **baselineとの比較（事実）**：P@10は人気順が最も高く、NCF・MF・平均評価を上回った。λ=0のNCF・MF・平均評価はrandomの14倍以上で、学習した成分はある。RMSEは平均評価（0.989）が最良で、MF（1.028）とNCF（1.061）は悪い。NCFはMFより悪い（検証データのval RMSEではMF・NCFが平均評価より良かったため、評価の設定によって順位が変わった）。
  - **仮説（未検証）**：5本の入力だけのfold-inが過剰にフィットしているのではないか。fold-inの設定（l2・step数・lr）はdevだけで調整できるので、10/30の改善候補とする。
  - **限界**：差の統計的な裏づけ（信頼区間）は計算していない。MovieLensは評価済みの映画だけが正解で、P@10の絶対値は低く出る。入力が5本の設定での結果であり、他の入力本数には当てはまらない。
  - **不具合の修正（7章）**：順位づけがclip後の値だった不具合を修正した。修正前後で、MFのP@10は0.2035→0.2126、NCF（λ=0）は0.1735→0.1781とわずかに上がった。RMSEと他の手法は変わらない。人気順との差は縮まらなかった。

---

## 7. 開発記録の候補

`DEV_LOG.md` に書く候補を記録しておく。

| 日付 | AIへの依頼 | 問題 | 確認方法 | 修正 |
|---|---|---|---|---|
| 10/24 | 全dev（604人）の評価script（`evaluate.py`）をCopilotに依頼 | `--limit 50` では動いたが、全devで `ValueError`（映画ID 133が学習データの対応表にない） | 全devの実行でエラーを再現。dev利用者の入力に、train利用者の評価がなく対応表に無い映画が含まれていた | 評価側で対応表外の入力映画を除いてfold-inし、3本未満の利用者は全モデルで対象外にした（結果：延べ2件・2人、対象外0人）。`recommender.py` は変更せず、アプリでは拒否のまま |
| 10/24 | アプリで高評価ばかりを入力して動作確認 | 予測評価が5.00で同点だらけ。生の予測は最大5.54で、5.0以上が67本（アニメ5本の入力） | 一時スクリプトで生の予測を表示。`recommend()` と `evaluate.py` が、clip後の値で順位づけしていた（同点はmovie_id昇順） | 順位づけをclip前の値に変更し、clipは表示とRMSEだけにした。テスト2件を追加（計23件）。devを再実行して前後を比較 |

---

## 8. 残りの作業計画（10/26〜11/3）

| 期間 | 作業 | 完了条件 |
|---|---|---|
| 10/22〜10/25（提出まで） | 分割を固定（10/24。testデータの確定日を兼ねる）（seed、assert）。MovieLens読込、平均評価・行列分解、Streamlit画面。NCF学習とfold-in。動作確認表と初期結果（dev）。commit | 入力から出力まで画面で通る。本ファイルの実測欄がすべて埋まる |
| 10/26〜10/29 | 10/26：dev全指標（平均・人気・MF・NCF）。10/27：λの比較（dev）、既定λを決定。10/28：**test 1回目**（全手法・全λ）。10/29：失敗例を3件以上集め、各々の原因段階を1つ選ぶ | 条件別の結果の表（dev・test）。失敗例ごとに根拠つきの原因 |
| 10/30〜11/2 | 10/30：改善を1つ試す（devのみ）。11/1：**test 2回目**。11/1〜2：`REPORT.md`、`DEV_LOG.md`（重要な修正3件）、README・requirements.txt、デモ動画（成功例・失敗例を1件以上ずつ）。11/2：新しい環境でREADME通りに動くか確認 | 提出物がそろい、README通りに起動できる |
| 11/3 | 最終提出 | repository・`REPORT.md`・`DEV_LOG.md`・デモ動画 |

- 止まったときの縮小案（11/2までに）：fold-inが動かなければ「入力映画の埋め込み平均との類似度」に切り替える（企画書の代替案）。NCFが行列分解に勝てなくても、そのまま報告して差の理由を分析する。余裕がなければ人気順の追加と改善の試行を省く。
- 評価・比較・失敗分析の配点が最も大きい（30点）ため、機能追加より、dev/testの分離と条件比較を優先する。

---

## 提出前の確認

- [x] 企画からの変更点と理由を書いた
- [x] 処理の流れで、深層学習の部分とルールの部分を分けた
- [x] モデルの入出力を、実際に表示した値で確認した
- [x] 入力から出力までの動作確認を、対象外の入力も含めて記録した
- [x] devとtestを分け、testを最後まで使わない方法を書いた
- [x] 指標・比較対象・比較する条件を書いた
- [x] 残りの作業計画を書いた
- [x] repository直下に `PROTOTYPE_EVAL_PLAN.md` として保存し、commitした
