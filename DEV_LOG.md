# 深層学習アプリ開発課題 開発記録

- 学籍番号：2432059
- 氏名：平田琉翔
- 提出file名：`DEV_LOG.md`（repository直下に置き、commitして提出する）
- 締切：**11/3（火）23:59**

---

## 使用したAIツール

- GitHub Copilot CLI：コードの作成・修正。先に計画を出させ、確認してから実装させた。git操作とtest評価の実行は、Copilotにさせない決まりにした。
- Claude：評価計画・結果の分析・修正方針の相談。

---

## 修正1：予測を1〜5に切り詰めた値で順位づけしていた問題（2026-10-24）

- 何を依頼したか（promptの要点）：
Copilotへの依頼は3段階（調査→計画→実装）。すべて、実際に送った文のまま。

（1) 調査。コードは変更させない）
```text
コードを変更せず、調べて報告してください。
1. recommender.py の predict_ratings / rerank / recommend で、順位づけに使う値は「クリップ前」か「クリップ後」か。該当する行を示す。
2. evaluate.py の Precision@10 の順位づけは、どの関数のどの値(クリップ前/後)を使っているか。同点のときの並びはどう決まるか。
3. 次の入力で、先頭30件について、クリップ前の予測値(生の値)を表示する一時スクリプトを scratch で実行し、結果を示す(リポジトリにファイルを残さない)。
   入力: Toy Story(1995)=5, Wallace & Gromit: The Best of Aardman Animation(1996)=5, Beauty and the Beast(1991)=5, Aladdin(1992)=5, Snow White and the Seven Dwarfs(1937)=4
   モデル: ncf。5.0 以上の映画が何本あるか、クリップ後に同点になっている映画が何本あるかも示す。
git の操作はしない。既存ファイルは変更しない。
```
（2) 計画。実装はさせない）
```text
次の不具合を修正したいです。まず計画だけ出してください(コードはまだ書かない)。

不具合: recommend() と evaluate.py(MF/NCF)の順位づけが、1〜5 にクリップした予測値で行われている。生の予測が 5 を超える映画が多数あり、5.0 で同点になって、同点は movie_id 昇順で並ぶため、本来の順位が失われる(Toy Story ほか5本の入力で、生の予測 5.0 以上が67本)。

方針:
- 順位づけ(rerank の入力)にはクリップ前の予測値を使う。rerank の式は 生の予測 + lam * 新しさ のまま。
- クリップは、(a) 画面などに返す表示用の予測評価、(b) RMSE の計算、にだけ使う。
- predict_ratings() の既定の挙動(クリップあり)は変えない。必要なら clip=True の引数を追加して、False のときは生の値を返すようにする。
- recommend() の返す予測評価(表示用)はクリップ後のまま。
- evaluate.py も、順位づけにはクリップ前の値、RMSE にはクリップ後の値を使う。mean / popular / random は変更しない。
- テストを追加する: 生の予測が 5 を超える2映画(例 5.4 と 5.2)が、λ=0 のときにクリップ後に同点にならず、順位が保たれること。

計画には、変更するファイルと関数、テストの内容、既存の21件のテストへの影響を書くこと。git の操作はしない。既存のテスト結果ファイルは変更しない。
```
（3) 実装。自分で2点を追加）
```text
計画どおりに実装して。ただし次を追加してください。
1. check_model.py の「生予測上位5件」は、predict_ratings(clip=False) の値で表示し、見出しに「クリップ前」と明記する。それ以外の check_model.py の出力は変えない。
2. predict_ratings() を呼んでいる他の箇所(app.py、check_model.py、evaluate.py、tests など)を grep で洗い出し、clip の既定値(True)に依存して壊れる箇所が無いことを確認して報告する。
変更するファイルは、recommender.py、evaluate.py、check_model.py、tests/test_recommender.py、tests/test_evaluate.py の5つだけ。git の操作はしない。既存の結果ファイル(results/ 以下)は変更しない。実装後は python -m pytest -q を実行して結果を示すこと。
```
- 何が問題だったか：アプリにアニメ5本（Toy Story=5、Wallace & Gromit: The Best of Aardman Animation=5、Beauty and the Beast=5、Aladdin=5、Snow White and the Seven Dwarfs=4）を入力すると、予測評価が 5.00 で同点になった。生の予測が 5.0 以上の映画が67本あり（最大 5.54）、`recommend()` と `evaluate.py` が1〜5に切り詰めた値で順位づけしていたため、同点が `movie_id` 昇順で並び、順位が失われていた。
- どう確認したか：画面で再現し、Copilotに調べさせて原因を確認した。修正前は上位10件のうち9件が 5.00 だった。
修正前の画面（2026-10-24 16:39、λ=0.25）：上位10件のうち9件が 5.00（残る1件は 4.96）。（画面の画像はリポジトリに含めない。値は上記の通り、記録（DEV_LOG_NOTES.md）にも残している）
- 自分で確認・判断した点：Copilotに、まずコードを変更させずに原因を調べさせた（順位づけに使う値、同点の並び、生の予測が5.0以上の本数）。計画を読んでから実装させ、実装の依頼には「check_model.py の出力見出しを『クリップ前』にする」「`predict_ratings()` を呼ぶ他の箇所を grep で洗い出し、既定値に依存して壊れる箇所がないことを報告する」を自分で追加した。
- どう修正したか：順位づけにはクリップ前の値を使い、クリップは表示とRMSEの計算だけにした（commit 69e2069）。テストを2件追加し、`pytest` は計23件が通った。dev 604人で再評価すると、RMSEと mean / popular / random は変わらず、Precision@10（上位25%基準）はわずかに上がった（MF 0.2035→0.2126、NCF λ=0 0.1735→0.1781）。人気順（0.296）との差は縮まらなかった。旧結果は `results/dev_all_default.txt`、新結果は `results/dev_all_rankfix.txt`。testは使っていない。

```diff
-    return torch.clamp(predictions, min=1.0, max=5.0).cpu().numpy()
+    if clip:
+        predictions = torch.clamp(predictions, min=1.0, max=5.0)
+    return predictions.cpu().numpy()
...
-    raw_predictions = _raw_predictions(model, user_params)
+    raw_predictions = predict_ratings(model, user_params, clip=False)
...
-        candidates.append((movie_id, float(predictions[index]), genres))
+        candidates.append((movie_id, float(raw_predictions[index]), genres))
```
- この修正から分かったこと：表示用の値と順位づけ用の値は分ける必要がある。画面で気づいた見た目の問題（5.00が並ぶ）の原因は、予測を出力する前の処理にあった。修正しても指標の改善は小さく、人気順に劣る問題は残った。

---

## 修正2：訓練データにない映画が入力に含まれ、全dev評価が止まった問題（2026-10-24）

- 何を依頼したか（promptの要点）：dev/testを評価する `evaluate.py` をCopilotに依頼した（`COPILOT_PROMPTS.md` の S10 の文をそのまま使用）。
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
- 何が問題だったか：`--limit 50` では全モデルが正常に動いたが、全dev（604人）の `--all` で `ValueError`（映画ID 133 は対応表にない）が出て止まった。devの利用者の入力5本に、train利用者の評価がなく、モデルの対応表にない映画が含まれていた。
- どう確認したか：全dev実行でエラーを再現した。`validate_inputs` は、アプリで使うために対応表にない映画を拒否する仕様だと確認した。
```text
ValueError: 映画ID 133 は対応表にない
（全dev実行時のtracebackは保存していない。エラーの内容は、当時のメモ DEV_LOG_NOTES.md の記録）
```
- どう修正したか：`recommender.py` は変更せず、評価側（`evaluate.py`）で対応表にない入力映画を除いてから fold-in するようにした。入力が3本未満になる利用者は、全モデルで対象外にした。実際に影響を受けたのは延べ2件（2人）で、対象外になった利用者は0人だった。除いた件数は、結果の表（「入力から除いた映画の延べ件数」「入力が減った利用者数」「入力3本未満で対象外にした利用者数」）に出す。
（commit 9ae7e80、`evaluate.py` の要点）
```diff
+    movie_to_index, _, _, _ = build_id_maps(training_data["ratings"])
+    for user_id, user_data in split_data.items():
+        original_input = user_data["input"]
+        used_input = [
+            (movie_id, rating)
+            for movie_id, rating in original_input
+            if movie_id in movie_to_index
+        ]
+        removed_count = len(original_input) - len(used_input)
+        ...
+        if len(used_input) < 3:
+            users_excluded_for_short_input += 1
+            continue
+        eligible_split_data[user_id] = {**user_data, "_used_input": used_input}
...
-        seen_ids = {int(movie_id) for movie_id, _ in input_items}
+        seen_ids = {int(movie_id) for movie_id, _ in original_input_items}
```
除いた映画も「見た映画」として扱い、推薦候補から外している（`seen_ids` は元の入力から作る）。
- この修正から分かったこと：アプリ用の入力検査をそのまま評価に使うと、評価データの事情に合わない場合がある。除外した件数も結果として記録する必要がある。

---

## 修正3：fold-in の正則化（l2）を、devだけで決めて変更した（2026-10-30）

- 何を依頼したか（promptの要点）：
Copilotへの依頼は2組（引数の追加、既定値の変更）。それぞれ「計画だけ → 計画どおりに実装して」の2段階。すべて、実際に送った文のまま。

（1-1) 引数の追加：計画）
```text
evaluate.py に、fold-in の設定を指定する引数を追加したいです。まず計画だけ出してください(コードはまだ書かない)。

要件:
- 引数 --fold-l2(既定0.01)、--fold-steps(既定50)、--fold-lr(既定0.05)を追加する。fold_in_user() に渡す。既定値のとき、既存の結果は変わらないこと(再現性)。
- 結果の表(標準出力と results/*.csv)に、使った fold-in の設定(l2, steps, lr)の列を追加する。
- test の保護(--all と確認プロンプト、results/test_runs.log への記録)は変更しない。
- recommender.py の fold_in_user() の既定値は変えない(evaluate.py 側から引数で渡す)。
- 既存のテスト(23件)に影響しないこと。引数追加のテストを1件追加する。

計画には、変更するファイルと関数、テストの内容、既存テストへの影響を書くこと。
git の操作はしない。test の評価(splits/test)は実行しない。yes を自動入力するコマンドも使わない。
```
（1-2) 引数の追加：実装。自分で2点を追加）
```text
計画どおりに実装して。ただし次を追加してください。
1. テストでは、parse_args() の確認に加えて、引数で指定した fold-l2 / fold-steps / fold-lr が、fold_in_user() を呼ぶときのキーワード引数に実際に渡ることを、モックの呼び出し記録で確認する。
2. results/test_runs.log に書く1行に、fold-in の設定(l2, steps, lr)も追記する。それ以外(確認プロンプト、--all の制限、ログの日時やモデル一覧)は変えない。
実装後、python -m pytest -q を実行して結果を示すこと。git の操作はしない。splits/test の評価は実行しない。yes を自動入力するコマンドは使わない。
```
（2-1) 既定値の変更：計画（devで採用した後））
```text
fold-in の既定の l2 を 0.01 から 1.0 に変更したいです。まず計画だけ出してください(コードはまだ書かない)。

背景: dev での実験の結果、l2=1.0 を採用した(docs/IMPROVEMENT_PLAN.md)。アプリ(app.py)も評価(evaluate.py)も、同じ既定値を使うようにしたい。

要件:
- recommender.py の fold_in_user() の既定 l2 を 1.0 にする。evaluate.py の --fold-l2 の既定も 1.0 にする。app.py と check_model.py は recommender の既定値を使う(個別に値を書かない)。
- 従来の結果を再現したいときは、--fold-l2 0.01 で再現できることを保つ。
- 既存のテストで、l2 の既定値 0.01 に依存しているものがあれば洗い出し、期待値を更新する計画を書く。
- fold_in_user の既定値が 1.0 であることを確認するテストを1件追加する。

git の操作はしない。splits/test の評価は実行しない。yes を自動入力するコマンドは使わない。
```
（2-2) 既定値の変更：実装。自分で2点を追加）
```text
計画どおりに実装して。ただし次を追加してください。
1. テストが失敗したとき、期待値を更新してよいのは、原因が「fold_in_user の既定 l2 の変更(0.01→1.0)」だと確認できたものだけ。それ以外の失敗は更新せず、内容を報告すること。更新した期待値は、どのテストの何をどう変えたか一覧で示すこと。
2. docs/DESIGN.md の fold-in の既定値(l2=0.01)の記述に、「既定値を 1.0 に変更した(理由は docs/IMPROVEMENT_PLAN.md)」と1行追記する。PROTOTYPE_EVAL_PLAN.md は変更しない。
変更するファイルは、recommender.py、evaluate.py、tests/test_evaluate.py、tests/test_recommender.py、docs/DESIGN.md の5つ。実装後、python -m pytest -q を実行して結果を示すこと。git の操作はしない。splits/test の評価は実行しない。yes を自動入力するコマンドは使わない。
```
- 何が問題だったか：devでNCFのRMSEが 1.0605 で、平均評価（0.9888）より悪かった。仮説は、「fold-in が5本の入力に過剰にフィットしている」こと。
- どう確認したか：結果を見る前に、採用基準を `docs/IMPROVEMENT_PLAN.md` に書いてcommitしてから、l2 を5水準でdev評価した（testは使っていない）。
```text
NCF（λ=0）のdev RMSE pooled
l2=0.01  1.060513
l2=0.03  1.044481
l2=0.1   1.019782
l2=0.3   0.990383
l2=1.0   0.96367     （MFは 1.0284 → 0.9444、平均評価は 0.9888）
```
- 自分で確認・判断した点：実装の依頼に、「引数が `fold_in_user()` に実際に渡ることをモックで確認する」「`test_runs.log` に fold-in の設定も記録する」を追加した。既定値の変更では、「期待値を更新してよいのは、原因が l2 の既定値の変更だと確認できたテストだけ」と条件をつけた。
- どう修正したか：基準を満たしたので、`--fold-l2` の既定値を 0.01 → 1.0 に変更した（commit 78942e8）。P@10 も上がった（NCF λ=0：0.1781→0.1879）。人気順（0.296）には届かない。

```diff
-    parser.add_argument("--fold-l2", type=float, default=0.01)
+    parser.add_argument("--fold-l2", type=float, default=1.0)
```
- 自分で確認・判断した点：「強いl2は、低評価寄りの入力の利用者に悪い」という自分の予想は、データで否定された（l2 を強めると、どのグループも改善した）。一方、低評価寄りの入力の利用者は、高評価寄りより悪いまま（NCF 1.090 と 0.886）。
- この修正から分かったこと：基準を先に決めておくと、結果を見てから基準を変えることを避けられる。ただし最良の値が実験範囲の端（1.0）にあり、傾向は続いているため、最適値は未確認。

---

## 提出前の確認

- [x] 3件それぞれに、依頼・問題・確認・修正を書いた
- [x] 確認の証拠（出力・test・値）を貼った
- [x] AIの提案を採用しなかった理由、または修正した理由を書いた（修正3：自分の仮説がデータで否定された点を記載。採用しなかった提案があれば追記）
- [x] API keyや個人情報を書いていない
