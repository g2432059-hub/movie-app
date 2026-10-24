# 実装手順マニュアル：映画推薦アプリ（第8回試作品）

- 作成日：2026-10-24
- 対象：`~/movie-app`（WSL2 / bash）で、GitHub Copilot CLI を使って実装する
- 締切：`PROTOTYPE_EVAL_PLAN.md` は **10/25（日）23:59**
- 関連：`PROTOTYPE_EVAL_PLAN.md`（提出物）、`DESIGN.md`（設計）、`COPILOT_PROMPTS.md`（promptの一覧）

> このマニュアルは、**上から順に、書いてあるコマンドをそのまま入力する**ことで完了するように書いてある。
> コマンド中の `<...>` だけは、自分の内容に置き換える。それ以外は変更しない。
> 「期待」と違う結果が出たら、次のstepへ進まず、末尾の「困ったとき」を見る。

## 0. 全体の流れと日程の目安

| step | 内容 | 目安 | 日 |
|---|---|---|---|
| S1 | repositoryを作る | 20分 | 10/24 |
| S2 | Python環境を作る | 20分 | 10/24 |
| S3 | データを取得する | 15分 | 10/24 |
| S4 | data_utils.py | 25分 | 10/24 |
| S5 | make_split.py | 30分 | 10/24 |
| S6 | metrics.py とテスト | 30分 | 10/24 |
| S7 | train.py（平均評価・行列分解） | 45分 | 10/24 |
| S8 | NCF を追加 | 35分 | 10/24〜25 |
| S9 | recommender.py（fold-in・再ランキング） | 60分 | 10/25 |
| S10 | evaluate.py と dev の評価 | 50分 | 10/25 |
| S11 | check_model.py | 20分 | 10/25 |
| S12 | app.py（画面）と動作確認 | 40分 | 10/25 |
| S13 | README・requirements | 20分 | 10/25 |
| S14 | PROTOTYPE_EVAL_PLAN.md を完成 | 40分 | 10/25 |
| S15 | 最終確認と提出 | 20分 | 10/25 |

- 合計の目安は約7時間半。Copilot の出力の修正で延びることが多い(推測)。
- **S9 の fold-in が 10/25 の昼までに動かない場合は、S9 の「代替(fallback)」に切り替える。** 動いた範囲で提出してよい(`PROTOTYPE_EVAL_PLAN.md` の6章に「動いた範囲」を書く)。

## 1. 共通の作法

1. **1つのstepが終わるごとに、必ずcommitする。** commitの手順は各stepに省略せず書いてある。
2. Copilot には**必ず変更計画を先に出させ**、自分で読んでから実装させる。
3. Copilot が書いたcodeは、**自分で実行して確かめる**(各stepの【確認】)。
4. Copilot にgit操作(add/commit/push)はさせない。
5. `data/`（MovieLens）と `models/`（学習済み重み）と `dla/`（仮想環境）はgit管理しない。
6. **testの評価は2回まで**(10/28と11/1の予定)。それ以外で `splits/test` を指定した `evaluate.py` を実行しない。確認の質問が出たら `no` と答える。
7. 新しいterminalを開いたら、毎回 `cd ~/movie-app` と `source dla/bin/activate` を入力する。

---

## S1　repositoryを作る

- 目安時間：20分

【実行】作業用のfolderを作り、gitを初期化する。

```bash
mkdir -p ~/movie-app
cd ~/movie-app
pwd
ls -la
[ -d .git ] || git init -b main
git status
git branch --show-current
git log --oneline -5
git remote -v
git config user.name
git config user.email
```

- `[ -d .git ] || git init -b main` は、`.git` がなければ初期化し、すでにあれば何もしない(既存のrepositoryを壊さない)。
- 出力を見て、状態を次のどちらかに分ける。
  - **新規(fileがほぼない)**：そのまま次へ進む。
  - **既存(app.py、README.md、requirements.txt、gitの履歴などがある)**：消したり上書きしたりせず、そのまま次へ進む。既存のfileは、後のstepでCopilotに「既存の内容を確認し、何を残し何を置き換えるかを計画に書く」よう指示してある。`git status` が clean でなければ、`git add -A` で一括commitせず、**`git diff` で内容を確認してから**commitする。
- `git branch --show-current` が `main` 以外(例：`feature/...`)なら、そのbranchのまま作業してよい。最終的にどのbranchを提出するかは、授業の指示に従う。
- `git remote -v` に `origin` がすでに表示されていれば、後の「GitHubにremoteを作る」手順の2〜4は不要(1回目のpushのみ行う)。
- `git config user.name` と `git config user.email` が空なら、次を入力する(`<...>` は自分の名前とメールアドレスに置き換える)。すでに表示されたらこの手順は不要。

```bash
git config --global user.name "<自分の名前>"
git config --global user.email "<自分のメールアドレス>"
```

期待：`pwd` が `/home/ryuto/movie-app` と表示される。

【実行】`.gitignore` と、必要な空folderを作る。

```bash
cd ~/movie-app
touch .gitignore
for L in dla/ data/ models/ __pycache__/ .pytest_cache/ '*.pyc'; do grep -qxF "$L" .gitignore || echo "$L" >> .gitignore; done
mkdir -p docs tests splits results data
cat .gitignore
ls -a
```

期待：`.gitignore` に `dla/` `data/` `models/` `__pycache__/` `.pytest_cache/` `*.pyc` の6つが(既存の行があればそれに加えて)含まれ、`ls -a` に `docs data results splits tests` が出る。

【実行】Windows側の授業folderから、資料をrepositoryへコピーする。

```bash
SRC="/mnt/c/Users/ryuto/OneDrive/ドキュメント/授業ファイル/深層学習応用事例"
ls "$SRC/第8回1022"
ls "$SRC/提出用フォルダー"
cp "$SRC/提出用フォルダー/IDEA_PROPOSAL.md" ~/movie-app/IDEA_PROPOSAL.md
cp "$SRC/第8回1022/PROTOTYPE_EVAL_PLAN.md" ~/movie-app/PROTOTYPE_EVAL_PLAN.md
cp "$SRC/第8回1022/DESIGN.md" ~/movie-app/docs/DESIGN.md
cp "$SRC/第8回1022/COPILOT_PROMPTS.md" ~/movie-app/docs/COPILOT_PROMPTS.md
cp "$SRC/第8回1022/IMPLEMENTATION_MANUAL.md" ~/movie-app/docs/IMPLEMENTATION_MANUAL.md
mkdir -p ~/movie-app/docs/templates
unzip -o "$SRC/第8回1022/lesson08.zip" -d ~/movie-app/docs/templates
cd ~/movie-app
ls -R docs | head -30
ls
```

- `unzip` がないと言われたら `sudo apt update && sudo apt install -y unzip` を入力して、もう一度 `unzip` の行から実行する。
- `/mnt/c/...` が見つからない場合は、Windows側のパスが違う。`ls /mnt/c/Users/` で自分のuser名を確認し、`SRC=` の行を直す。

期待：repository直下に `IDEA_PROPOSAL.md` と `PROTOTYPE_EVAL_PLAN.md` があり、`docs/` に `DESIGN.md` などがある。

【実行】Copilotに渡す前提ruleのfile `CONTEXT.md` と、開発記録のメモ用fileを作る。

```bash
cd ~/movie-app
cat > CONTEXT.md <<'EOF'
# プロジェクトの前提(Copilotへ)

- 内容: 映画推薦アプリ。見た映画3本以上と評価(1〜5)を入力すると、おすすめ10本(映画名・予測評価・ジャンル)を表示する
- 詳細設計: docs/DESIGN.md、評価計画: PROTOTYPE_EVAL_PLAN.md

## ルール
- code を書く前に変更計画(file、関数名、引数、戻り値)を示す。私が承認するまで実装しない
- 頼んでいないfileの変更・機能追加・リファクタリングをしない
- gitの操作(add/commit/push)は行わない。私が行う
- test(splits/test_*)の中身を読む・集計する・表示するcodeを、evaluate.py と make_split.py 以外に書かない。学習や調整にtestを使わない
- 乱数seedは42
- 画面(app.py)は推論(recommender.py)を呼ぶだけにする
- 関数にはdocstring(入力・戻り値の形)を書く。コメントは日本語
- 依存は numpy, pandas, torch, streamlit, pytest のみ。追加が必要なら理由を示して確認する
- data/ と models/ はgit管理しない

## 評価の定義
- RMSE: pooled(評価用の全評価をまとめる)を主、利用者ごとのRMSEの平均を副。予測は1〜5にclip。入力の5本は評価に含めない
- 高評価: 主 = その利用者の評価用データの75パーセンタイル以上(numpy.percentile(..., 75, method="lower")、同点を含む)。評価用の評価が全部同じ値の利用者は主指標から除外し、人数を報告する。副 = 4点以上
- Precision@10: 候補 = 入力以外の全映画(trainに無い映画は除く)。上位10件のうち高評価の割合。利用者ごとに計算して平均
- ジャンル数: 上位10件に含まれるジャンルの種類(重複なし)
- 再ランキング: 予測評価 + λ × (未選択ジャンルの割合)。同点は映画ID昇順

## データ
- data/ml-1m/ratings.dat, movies.dat (区切りは "::")
- 分割: 利用者単位で train80% / dev10% / test10% (seed42)。dev/testは各利用者の評価から5本を入力用、残りを評価用
- early stopping は train 利用者の評価の5%を検証用(val)として取り分けて行う
EOF
cat > DEV_LOG_NOTES.md <<'EOF'
# 開発記録のメモ(DEV_LOG.md を書くときの材料)
EOF
wc -l CONTEXT.md DEV_LOG_NOTES.md
```

期待：`CONTEXT.md` が約30行、`DEV_LOG_NOTES.md` が1行。

【実行】GitHubにremoteを作る(任意だが、バックアップと提出のために推奨)。

1. ブラウザで https://github.com/new を開く。
2. Repository name に `movie-app` と入力する(すでにGitHubにある場合はこの手順は不要)。**Private** を選ぶ。README・.gitignore・licenseは**追加しない**。
3. 「Create repository」を押す。
4. 次を入力する。

```bash
cd ~/movie-app
git remote get-url origin || git remote add origin https://github.com/g2432059-hub/movie-app.git
git remote -v
```

期待：`origin` が2行(fetch/push)表示される。GitHubのuser名が違う場合は、URLの `g2432059-hub` を自分のuser名にする。

【commit】3回に分ける。

1回目：

```bash
cd ~/movie-app
git status
git add .gitignore
git status
git diff --staged --stat
git commit -m "Add gitignore"
git log --oneline -3
```

2回目：

```bash
cd ~/movie-app
git status
git add IDEA_PROPOSAL.md PROTOTYPE_EVAL_PLAN.md docs
git status
git diff --staged --stat
git commit -m "Add proposal, evaluation plan, design docs"
git log --oneline -3
```

3回目(初回のpushも行う)：

```bash
cd ~/movie-app
git status
git add CONTEXT.md DEV_LOG_NOTES.md
git status
git diff --staged --stat
git commit -m "Add Copilot context and dev log notes"
git log --oneline -5
git push -u origin "$(git branch --show-current)"
```

- 初回のpushで認証を求められたら、第2回で設定した方法(ブラウザ認証やtoken)で認証する。**tokenやpasswordをfileに書かない。**
- 期待：`git log --oneline -5` に3つのcommitが並ぶ。`git push` が成功し、GitHubの画面にfileが表示される。
- 2回目の `git status` で `data/` が出ないのは、`.gitignore` で無視されているため(正しい)。

---

## S2　Python環境を作る

- 目安時間：20分

【実行】Pythonの版を確認し、仮想環境 `dla` を作る。

```bash
cd ~/movie-app
python3 --version
python3 -m venv dla
source dla/bin/activate
which python
python --version
python -m pip install --upgrade pip
```

- `python3 -m venv dla` で `ensurepip is not available` と出たら、`sudo apt update && sudo apt install -y python3-venv` を入力し、`rm -rf dla` のあと `python3 -m venv dla` からやり直す。
- 期待：`which python` が `/home/ryuto/movie-app/dla/bin/python` を指す。行頭に `(dla)` が付く。
- すでに別の場所に `dla` があり、torch などが入っている場合も、この手順で新しく作ってよい(混乱を避けるため)。

【実行】必要なlibraryを入れる。CPU版のtorchを使う(WSLにGPUを使わなくても動かすため)。

```bash
cd ~/movie-app
source dla/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas streamlit pytest
python -c "import torch, numpy, pandas, streamlit; print(torch.__version__, numpy.__version__, pandas.__version__, streamlit.__version__); print(torch.cuda.is_available())"
```

- 期待：4つの版番号が表示され、最後に `False`(CPU版のため)。
- torchのinstallに失敗したら、`python3 --version` の版が新しすぎる可能性がある(要確認)。「困ったとき」を見る。

【実行】依存を `requirements.txt` に記録し、`pytest` が動くことを確認する。

```bash
cd ~/movie-app
source dla/bin/activate
pip freeze > requirements.txt
wc -l requirements.txt
grep -i -E '^(torch|numpy|pandas|streamlit|pytest)==' requirements.txt
pytest --version
```

- 期待：`grep` で5つのlibraryが `==版番号` つきで出る。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add requirements.txt
git status
git diff --staged --stat
git commit -m "Add requirements"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S3　データを取得する

- 目安時間：15分

【実行】MovieLens 1M をダウンロードして展開する。データは**gitに入れない**(`data/` は無視されている)。

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

```bash
cd ~/movie-app
source dla/bin/activate
curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
ls -l data/ml-1m.zip
unzip -o data/ml-1m.zip -d data
ls -l data/ml-1m
wc -l data/ml-1m/ratings.dat data/ml-1m/movies.dat data/ml-1m/users.dat
file -i data/ml-1m/movies.dat data/ml-1m/ratings.dat
head -3 data/ml-1m/ratings.dat
head -3 data/ml-1m/movies.dat
git status
```

期待(公式の値と私の記憶による。違う場合は先へ進まず「困ったとき」を見る)：

```text
ratings.dat   1000209 行
movies.dat    3883 行
users.dat     6040 行
ratings.dat の例:  1::1193::5::978300760
movies.dat の例:   1::Toy Story (1995)::Animation|Children's|Comedy
git status に data/ が出ない(無視されている)
```

- `file -i` の文字コードが `utf-8` ではなく `iso-8859-1` と出る可能性がある。出た文字コードを**メモ**する(S4のCopilotの計画で、同じ文字コードを使っているか確認する)。
- ダウンロードが失敗した場合は、ブラウザで https://grouplens.org/datasets/movielens/1m/ を開き、`ml-1m.zip` を手動でダウンロードして `data/` に置いてから `unzip` の行を実行する。

【実行】licenseの文言を確認する。

```bash
cd ~/movie-app
cat data/ml-1m/README | head -60
```

- 研究・教育目的での利用条件、再配布の禁止、出典の書き方が書いてある。内容を自分で読み、`PROTOTYPE_EVAL_PLAN.md` の3章のlicenseの欄(【要確認】の部分)に、読んだ内容を書く(S14で行う)。

【commit】このstepはgit管理するfileがない。次のコマンドで、何もcommit対象がないことを確認する。

```bash
cd ~/movie-app
git status
git log --oneline -3
```

期待：`nothing to commit, working tree clean`。

---

## S4　data_utils.py を作る

- 目安時間：25分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画(関数名・引数・戻り値)を先に示し、私が承認するまで実装しない。
data_utils.py を作る。
- load_ratings(path): data/ml-1m/ratings.dat を読み、列 user_id, movie_id, rating, timestamp の DataFrame を返す。区切りは "::"
- load_movies(path): data/ml-1m/movies.dat を読み、列 movie_id, title, genres(list[str]) の DataFrame を返す。ジャンルは "|" 区切り。文字コードは実際のfileに合わせ、どう確かめたかも示す
- build_id_maps(ratings_train): train の評価に現れる movie_id だけに連番index(0始まり)を振り、movie_id→index と index→movie_id の dict を返す。user_id も同様
- 全関数に docstring(戻り値の列名と型)を書く
- python data_utils.py で、ratings の行数、movies の行数、先頭5行を表示する __main__ を付ける
```
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
source dla/bin/activate
python data_utils.py
git status
```

期待：ratingsの行数が **1000209**、moviesの行数が **3883**、先頭5行が表示され、`movie_id` `rating` `genres` の列がある。genresは `['Animation', "Children's", 'Comedy']` のようなlistになっている。映画名の文字化け(`?` や変な記号)がない。

文字化けがある場合は、S3でメモした文字コードと、`data_utils.py` のencoding指定が一致しているか確認する。一致していなければ、Copilotに「文字コードを `<メモした値>` にして」と指示する。

```bash
cd ~/movie-app
grep -n -i 'encoding' data_utils.py
grep -n -i 'test' data_utils.py
```

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add data_utils.py
git status
git diff --staged --stat
git commit -m "Add data loading and id map utilities"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S5　make_split.py を作る(train/dev/test の分割)

- 目安時間：30分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
source dla/bin/activate
python make_split.py
ls -l splits
wc -l splits/train_users.txt splits/dev_users.txt splits/test_users.txt
```

期待：人数が **4832 / 604 / 604** (`wc -l` の行数と一致)。

3集合が重ならず、合計が全利用者になることを確認する。

```bash
cd ~/movie-app
cat splits/train_users.txt splits/dev_users.txt splits/test_users.txt | sort | uniq -d | wc -l
cat splits/train_users.txt splits/dev_users.txt splits/test_users.txt | sort -u | wc -l
```

期待：1行目が **0**(重なりなし)、2行目が **6040**。

同じコマンドを再実行して、同じfileができることを確認する。

```bash
cd ~/movie-app
source dla/bin/activate
sha256sum splits/* > /tmp/split_a.txt
python make_split.py
sha256sum splits/* > /tmp/split_b.txt
diff /tmp/split_a.txt /tmp/split_b.txt && echo SAME
```

期待：`SAME` と表示される。

入力用が5本ずつになっていることを確認する(devのみ。testの中身は見ない)。

```bash
cd ~/movie-app
source dla/bin/activate
python -c "import json; d=json.load(open('splits/dev_inputs.json')); print(len(d), set(len(v['input']) for v in d.values()), min(len(v['eval']) for v in d.values()))"
```

期待：`604 {5} 15` のように、人数604、入力の件数が `{5}`、評価用の最小件数が15以上(ML-1Mは1人20件以上のため)。

コードを読み、testの扱いを確認する。

```bash
cd ~/movie-app
grep -n -i 'test' make_split.py
```

- `grep` の結果に、testの評価値を集計・表示する処理がないこと。ファイルへの保存と人数の表示だけならよい。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add make_split.py splits
git status
git diff --staged --stat
git commit -m "Add user-level train/dev/test split"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S6　metrics.py とテストを作る

- 目安時間：30分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
source dla/bin/activate
pytest -q tests/test_metrics.py
grep -n -E '0\.6455|\[5, ?5, ?4|== ?4|== ?6|== ?3' tests/test_metrics.py
```

期待：`6 passed` 程度(テストの個数はCopilotの書き方で変わってよい)。`failed` や `error` がない。

`grep` で、**私が示した期待値(0.6455、閾値4、件数6、ジャンル数3)がテストに書かれている**ことを確認する。期待値がcodeの出力をそのまま写したものになっていないか、テストfileを自分で読む。

テストが通らないときは、**テストの期待値を変えず**、metrics.py のほうを直させる(上のpromptの数値は手計算で確かめてある)。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add metrics.py tests/test_metrics.py
git status
git diff --staged --stat
git commit -m "Add evaluation metrics with hand-computed tests"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S7　train.py（平均評価・行列分解）を作って学習する

- 目安時間：45分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

学習の前に、testを読んでいないことをコードで確認する。

```bash
cd ~/movie-app
grep -n -i -E 'test|dev' train.py models.py
```

期待：testやdevのfileを読む処理がない。コメントや変数名に出るだけならよい。

平均評価モデルを作る。

```bash
cd ~/movie-app
source dla/bin/activate
python train.py --model mean --shrink 10
ls -l models
```

行列分解を学習する(数分〜十数分)。

```bash
cd ~/movie-app
source dla/bin/activate
python train.py --model mf --dim 32 --epochs 20 2>&1 | tee results/train_mf.log
ls -l models
tail -5 results/train_mf.log
```

期待：

```text
- epochごとにtrain損失とval RMSEが表示される
- train損失が下がっていく(val RMSEは途中で下げ止まり、early stoppingで止まってよい)
- val RMSE が、平均評価の val RMSE より小さい(平均評価のRMSEは約1付近のはず。要確認)
- models/ にMFの重みとID対応表などが保存される
```

- 時間がかかりすぎる(15分以上)場合は、`--epochs 10` に減らして再実行する。

- val RMSE が下がらず、最初から1を大きく超える場合は、学習率(`--lr`)を `1e-3` から `5e-3` や `5e-4` に変えて再実行する。変えた値は `DEV_LOG_NOTES.md` に記録する。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add models.py train.py results/train_mf.log
git status
git diff --staged --stat
git commit -m "Add mean and matrix factorization training"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S8　NCF を追加して学習する

- 目安時間：35分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
models.py と train.py に NCF を追加する(既存のMFとmeanの動作は変えない)。
- NCF: 利用者埋め込み(dim)と映画埋め込み(dim)を連結(2*dim)→ 64 → 32 → 16 → 1 のMLP(中間層はReLU、出力は活性化関数なし)。予測 = 全体平均 + MLPの出力。Dropoutは引数 --dropout(既定0.1)
- fold-in用に、利用者ベクトルを外から与えて予測する関数 predict_with_user_params(user_vec, item_idx) を用意する
- train.py に --model ncf を追加。損失はMSE、Adam、valのRMSEでearly stopping
- 保存するもの: MFと同じ形式(重み、ID対応表、全体平均、train利用者ベクトルの平均など)
- 各epochの train損失と val RMSE を表示する
```
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
source dla/bin/activate
python train.py --model ncf --dim 32 --epochs 30 2>&1 | tee results/train_ncf.log
ls -l models
tail -5 results/train_ncf.log
```

期待：epochごとにtrain損失とval RMSEが表示され、train損失が下がる。valのRMSEが平均評価より小さい。

確認：MFとmeanが壊れていないこと。

```bash
cd ~/movie-app
source dla/bin/activate
python train.py --model mean --shrink 10
python train.py --model mf --dim 32 --epochs 3
git status
```

期待：エラーなく動く(MFは3epochだけで確認)。`git status` で `models/` が出ない。

NCFのval RMSEがMFより悪くても、**そのまま記録する**(MLPが内積に劣る場合があることが知られている。要確認)。無理に調整しない。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add models.py train.py results/train_ncf.log
git status
git diff --staged --stat
git commit -m "Add NCF training"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S9　recommender.py を作る(fold-in・再ランキング)

- 目安時間：60分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
source dla/bin/activate
pytest -q
```

期待：`test_metrics.py` と `test_recommender.py` がすべて `passed`。

rerank のテストが、**私が示した手計算の例**(lam=0 で A,B。lam=1 で A,C)になっているかを、テストfileを読んで確認する。

```bash
cd ~/movie-app
grep -n -E 'lam ?= ?(0|1)' tests/test_recommender.py
```

fold-in が動くか、次の短いscriptで確認する(1行で入力する)。

```bash
cd ~/movie-app
source dla/bin/activate
python - <<'EOF'
import json
from recommender import load_model, recommend
d = json.load(open('splits/dev_inputs.json'))
uid = next(iter(d))
items = [tuple(x) for x in d[uid]['input']]
print('input:', items)
print(recommend(items, lam=0.25, model_name='ncf', debug=True))
EOF
```

期待：debug情報(index配列のshapeとdtype、fold-inの最初と最後の損失)が表示され、**最後の損失が最初より小さい**。おすすめ10件が表示され、入力した映画は含まれない。

同じ入力で2回実行して、同じ結果になること(seed固定)も確認する。

【代替(fallback)】fold-inが動かない、または最後の損失が下がらず原因が分からない場合、**時間を決めて(例：30分)**切り替える。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

```text
CONTEXT.md と docs/DESIGN.md を読むこと。変更計画を先に示し、私が承認するまで実装しない。
recommender.py に、fold-in の代替方法を追加する(既存の関数は変えない)。
- fold_in_embedding_average(model, item_idx, ratings): 入力映画の学習済み映画ベクトルを、評価(1〜5)から全体平均を引いた値で重みづけして平均し、利用者ベクトルとする。これをNCFまたはMFの利用者ベクトルとして使い、predict_ratings で全映画を予測できるようにする
- recommend() に引数 fold_in_method ("optimize" または "average"、既定 "optimize") を追加する
- tests/test_recommender.py に、average方式でも ValueError の検証と rerank が変わらないことを確認するテストを追加する
```
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

切り替えた場合は、`PROTOTYPE_EVAL_PLAN.md` の2章と6章に「fold-inの代わりに埋め込み平均を使った」と書き、`DEV_LOG_NOTES.md` に理由を記録する。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add recommender.py tests/test_recommender.py
git status
git diff --staged --stat
git commit -m "Add recommender with fold-in and diversity reranking"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S10　evaluate.py を作って dev を評価する

- 目安時間：50分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

まず少人数(50人)で動作を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
python evaluate.py splits/dev --model mean --limit 50
python evaluate.py splits/dev --model random --limit 50
python evaluate.py splits/dev --model mf --limit 50
python evaluate.py splits/dev --model ncf --lam 0 --limit 50
```

期待：それぞれ、RMSE(pooled/macro)、Precision@10(主・副)、ジャンル数、除外人数、除外されたtrainに無い映画の件数が表示される。`randomのPrecision@10` が他より低い。

**testの保護を確認する。** 次の2つのコマンドを入力する。1つ目は拒否され、2つ目は確認の質問が出るので **`no`** と入力する。`yes` と入力しないこと。

```bash
cd ~/movie-app
source dla/bin/activate
python evaluate.py splits/test --model mean
python evaluate.py splits/test --all
```

期待：1つ目は「testは --all でだけ実行できます」と表示されて終了する。2つ目は「testの評価は2回までです。続けますか? (yes/no)」と出る。`no` で終了し、`results/test_runs.log` が作られない(または追記されない)。

```bash
cd ~/movie-app
ls results
cat results/test_runs.log 2>/dev/null || echo 'test_runs.log なし(正しい)'
```

全dev(604人)で評価する。λの尺度を見るため、λを変えて実行する。

```bash
cd ~/movie-app
source dla/bin/activate
python evaluate.py splits/dev --model mean 2>&1 | tee results/dev_mean.txt
python evaluate.py splits/dev --model popular 2>&1 | tee results/dev_popular.txt
python evaluate.py splits/dev --model random 2>&1 | tee results/dev_random.txt
python evaluate.py splits/dev --model mf --lam 0 2>&1 | tee results/dev_mf_lam0.txt
for L in 0 0.1 0.25 0.5 1.0; do python evaluate.py splits/dev --model ncf --lam $L 2>&1 | tee results/dev_ncf_lam$L.txt; done
```

期待：人数が604(主指標では評価が全部同じ値の利用者を除いた人数も表示される)。**λを上げるとジャンル数が増える**ことが多い(予想。実測で確認する)。逆の結果でも、そのまま記録する。

λの水準で順位がほとんど変わらない場合(λ=1でもジャンル数がλ=0とほぼ同じ)は、水準を見直す必要がある。`DEV_LOG_NOTES.md` に記録し、次の水準(例：0, 0.5, 1, 2, 4)を試してよい(devの尺度確認のみ)。

- `results/*.csv` や `results/*.txt` が作られる。内容を見て、`PROTOTYPE_EVAL_PLAN.md` の6章に使う。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add evaluate.py results
git status
git diff --staged --stat
git commit -m "Add evaluation script and initial dev results"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S11　check_model.py を作る(入出力の確認)

- 目安時間：20分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
source dla/bin/activate
python check_model.py 2>&1 | tee results/check_model_output.txt
```

期待：入力、index配列のdtype・shape、評価のmin/max、fold-inの最初と最後の損失、生の出力の上位5件、最終結果の上位5件、所要時間(秒)、実行環境(CPU)が表示される。

生の出力(clip前)と、最終結果の予測評価が対応しているか(clipとλの加点で値が違う場合は、その理由を自分の言葉で説明できるか)を確認する。

この出力は `PROTOTYPE_EVAL_PLAN.md` の3章に**そのまま貼る**(S14)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add check_model.py results/check_model_output.txt
git status
git diff --staged --stat
git commit -m "Add model input-output check script"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S12　app.py（画面）を作って動作確認する

- 目安時間：40分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【実行】画面を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
streamlit run app.py
```

- Windowsのブラウザで http://localhost:8501 を開く(自動で開かない場合も多い)。
- 止めるときは、terminalで Ctrl+C を押す。

【確認】`PROTOTYPE_EVAL_PLAN.md` の4章の表に、次の5つの入力の結果を**画面に表示されたとおり**に記録する(スクリーンショットも撮っておく。デモ動画の材料になる)。

| No | 入力 | 事前の期待 |
|---|---|---|
| 1 | 同じジャンル(例：SF)の高評価5本 | SF系が多く上位に出る |
| 2 | 異なるジャンル5本(高評価・低評価が混在) | 低評価の映画と近い作品が下がる |
| 3 | 2本だけ | エラー表示、推薦しない |
| 4 | データにない映画/評価6・0 | そもそも選べない、または拒否される |
| 5 | 5本すべて5点 | 人気作に偏る可能性。結果を記録 |

- 操作のたびにモデルを読み直して遅くならないか(1回の表示が数秒以内か)を確認する。
- 期待と違う結果が出たら、前処理・モデル・後処理のどこに原因がありそうかを1つ選び、根拠を `DEV_LOG_NOTES.md` に記録する(失敗分析の材料になる)。

【記録】Copilotの出力が間違っていた・自分で直した・計画を直させた、のいずれかがあった場合だけ、次を実行して `DEV_LOG_NOTES.md` に追記する(`<...>` は自分の言葉に置き換える)。

```bash
cd ~/movie-app
cat >> DEV_LOG_NOTES.md <<'EOF'

## <日付> <step名>
- 依頼: <Copilotに何を頼んだか>
- 問題: <期待と実際>
- 確認: <どのコマンドで何を確かめたか>
- 修正: <どう直したか>
EOF
```
追記したら、次のcommitに `DEV_LOG_NOTES.md` も含める(commit手順の `git add` の行に追加する)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add app.py
git status
git diff --staged --stat
git commit -m "Add Streamlit app"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S13　README と requirements を整える

- 目安時間：20分

【開始前】作業ディレクトリに入り、仮想環境を有効にし、状態を確認する。

```bash
cd ~/movie-app
source dla/bin/activate
git status
```
期待：行頭に `(dla)` が付く。`git status` は `nothing to commit, working tree clean`(前のstepでcommit済みの場合)。

【Copilot】Copilot CLI を起動する。

```bash
cd ~/movie-app
source dla/bin/activate
copilot
```
1. 次のpromptを**全文そのまま**貼り付けて送信する。

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
2. Copilot が出した**変更計画**を読む。次を確認し、1つでも外れたら「計画を直して」と指示する。
   - 作るfile名・関数名がpromptと一致している
   - 頼んでいないfileを変更する計画になっていない
   - testデータを読む処理が含まれていない(evaluate.py の指定部分を除く)
3. 問題なければ `計画どおりに実装して` と入力する。ファイル編集の許可を求められたら、内容を確認して許可する。
4. 完了したら、Copilot CLI を終了する(`/exit`。操作がバージョンで異なる場合は Ctrl+C を2回)。
5. **Copilot にgitのcommitはさせない。** commitは下の手順で自分で入力する。

【確認】

```bash
cd ~/movie-app
cat README.md
```

READMEの手順を**上から順に読み**、抜けがないか確認する(データの取得、環境構築、分割、学習、画面の起動、評価、testは2回まで)。

requirements.txt を更新する。

```bash
cd ~/movie-app
source dla/bin/activate
pip freeze > requirements.txt
git diff --stat requirements.txt
```

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add README.md requirements.txt
git status
git diff --staged --stat
git commit -m "Add README and update requirements"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S14　PROTOTYPE_EVAL_PLAN.md を完成させる

- 目安時間：40分

【実行】編集する。VS Code で開くか、nano で開く。

```bash
cd ~/movie-app
code PROTOTYPE_EVAL_PLAN.md
```

- `code` が使えなければ `nano PROTOTYPE_EVAL_PLAN.md` で開く。

【記入】次の順に、**【要実行】【記入】【提出時に記入】と書いてある箇所をすべて**実際の結果に置き換える。

| 章 | 記入するもの | 材料 |
|---|---|---|
| 冒頭 | 提出日 | 提出する日付 |
| 冒頭 | 「この版は下書きである」の注意書き | 削除する |
| 1章 | 企画からの変更点 | 採用した変更だけ残す。採用しない行は削除する |
| 3章 | licenseの【要確認】 | S3で読んだ `data/ml-1m/README` の内容 |
| 3章 | 実行環境と推論時間 | `results/check_model_output.txt` |
| 3章 | モデルへ渡す直前と出力 | `results/check_model_output.txt` を貼る |
| 4章 | `git log --oneline` | 下のコマンドの出力 |
| 4章 | 動作確認の表 | S12の5つの結果 |
| 5章 | dev/testの人数、評価実行の日時 | `wc -l splits/*_users.txt` と `results/` |
| 6章 | 初期結果(dev) | `results/dev_*.txt` の数値を表にする |
| 6章 | 気づいたこと | 実測から言えることだけ(推測は推測と書く) |
| 7章 | 開発記録の候補 | `DEV_LOG_NOTES.md` から3件 |
| 8章 | 作業計画 | 実際の進み具合に合わせて修正 |

```bash
cd ~/movie-app
git log --oneline
wc -l splits/train_users.txt splits/dev_users.txt splits/test_users.txt
ls results
```

【確認】未記入が残っていないか調べる。

```bash
cd ~/movie-app
grep -n -E '【要実行|【記入|【提出時|【要確認|【仮|【実行後|【devで' PROTOTYPE_EVAL_PLAN.md
```

期待：何も表示されない。表示された行は、実測値に置き換える。**ただし、NCFなどが間に合わなかった箇所は、「未実施」と正直に書き、8章に実施日を書く**(値を想像で埋めない)。

【確認】提出前のチェックボックスを `[ ]` から `[x]` にする(すべて実際に満たしてから)。

【commit】毎回この順で入力する(途中を省略しない)。

```bash
cd ~/movie-app
git status
git add PROTOTYPE_EVAL_PLAN.md DEV_LOG_NOTES.md
git status
git diff --staged --stat
git commit -m "Complete prototype evaluation plan"
git log --oneline -3
git push
```
- `git status` の2回目で、add したfileが `Changes to be committed` に並ぶこと。
- 意図しないfile(`data/`、`models/`、`dla/`、`__pycache__/`)が並んでいたら、commitせずに `git restore --staged <file>` で戻し、`.gitignore` を確認する。
- `git push` は、S1 でGitHubのremoteを設定した後に使える。remoteが未設定なら `fatal: No configured push destination` と出るが、commit自体は成功している。

---

## S15　最終確認と提出

- 目安時間：20分

【確認】repositoryを別の場所にcloneして、手順どおりに動くかを確かめる(新しい環境でも動くことの確認)。これは**時間があれば**行う(なければ次の【実行】へ)。

```bash
cd ~
rm -rf movie-app-check
git clone https://github.com/g2432059-hub/movie-app.git movie-app-check
cd movie-app-check
ls
cat README.md | head -40
```

- ここでは `ls` で必要なfileがあることを確認する(`PROTOTYPE_EVAL_PLAN.md`、`IDEA_PROPOSAL.md`、`README.md`、`requirements.txt`、各py)。
- 確認後、`cd ~ && rm -rf movie-app-check` で削除する。

【確認】repository直下に提出物があり、commit済みであること。

```bash
cd ~/movie-app
git status
ls
git log --oneline
git ls-files | head -40
```

期待：`git status` が `nothing to commit, working tree clean`。`git ls-files` に `PROTOTYPE_EVAL_PLAN.md` と `IDEA_PROPOSAL.md` が含まれ、`data/` `models/` `dla/` は含まれない。

【実行】提出用の印(tag)を付けてpushする。

```bash
cd ~/movie-app
git status
git tag -a prototype-1025 -m "Prototype and evaluation plan (lesson 8)"
git push
git push origin prototype-1025
git log --oneline --decorate -3
```

【実行】提出方法に合わせて提出物を用意する。**提出方法(GitHubのURL、zip、提出用フォルダーへのコピーなど)は、授業の指示(第7回までと同じ方法)に従う。** 私は提出先の指示を確認していない。

- zip を作る場合(Windowsの Downloads に出力する)：

```bash
cd ~/movie-app
git archive --format=zip --output=/mnt/c/Users/ryuto/Downloads/movie-app-prototype.zip HEAD
ls -l /mnt/c/Users/ryuto/Downloads/movie-app-prototype.zip
```

- 授業の提出用フォルダーへ `PROTOTYPE_EVAL_PLAN.md` をコピーする場合：

```bash
SRC="/mnt/c/Users/ryuto/OneDrive/ドキュメント/授業ファイル/深層学習応用事例"
cp ~/movie-app/PROTOTYPE_EVAL_PLAN.md "$SRC/提出用フォルダー/PROTOTYPE_EVAL_PLAN.md"
ls -l "$SRC/提出用フォルダー"
```

- 提出の締切は **10/25（日）23:59**。余裕をもって提出する。

【このあと(参考)】10/26〜11/3の予定は `PROTOTYPE_EVAL_PLAN.md` の8章と `DESIGN.md` の14章にある。次にやるのは、dev での条件比較(λ)、既定λの決定、test の1回目(10/28)である。testを実行するときは、実行前に `git status` が clean であることと、設定を変更しないことを確認し、`yes` と答える。

---

## 困ったとき

| 症状 | 原因の候補 | 対処 |
|---|---|---|
| `pip install torch` が失敗する | Pythonの版が新しすぎてtorchのwheelがない(要確認) | `python3 --version` を確認。`sudo apt install -y python3.11 python3.11-venv` 等で別の版を入れ、`rm -rf dla` のあと `python3.11 -m venv dla` から再開。それでも失敗するなら、エラー全文を記録して相談する |
| `python3 -m venv` が失敗する | `python3-venv` 未導入 | `sudo apt update && sudo apt install -y python3-venv` |
| `unzip: command not found` | 未導入 | `sudo apt update && sudo apt install -y unzip` |
| `ratings.dat` の行数が1000209でない | ダウンロード不完全 | `rm -rf data/ml-1m data/ml-1m.zip` してS3をやり直す |
| 映画名が文字化けする | 文字コードの指定違い | S3の `file -i` の結果に合わせる。Copilotに「`<文字コード>` で読むよう直して」と指示 |
| Copilotがtestを読むcodeを書いた | prompt/CONTEXT.mdの守られていない | 該当箇所の削除を指示し、`grep -n -i test <file>` で確認する。`DEV_LOG_NOTES.md` に記録(重要な修正の候補) |
| Copilotが勝手にcommitした | 許可を与えた | `git log --oneline` で確認。意図しないcommitは `git reset --soft HEAD~1` で戻す(変更は残る)。以後は許可しない |
| テストが通らない | codeの誤り(期待値は手計算) | **テストの期待値を変えず**、codeを直させる |
| 学習が遅い | CPUで重い設定 | `--epochs` と `--batch` を調整。それでも15分以上なら `--dim 16` |
| fold-inの損失が下がらない | lrが不適切、または実装誤り | `lr` を0.05→0.01、`steps` を50→100で試す(devの確認のみ)。30分で解決しなければS9のfallbackへ |
| `streamlit run` で画面が開かない | WSLからブラウザが自動で開かない | 表示された `http://localhost:8501` をWindowsのブラウザに手で入力 |
| `git push` が拒否される | remoteに別のcommitがある、または認証 | `git pull --rebase origin main` のあと `git push`。認証エラーは第2回の設定を見直す |
| `git commit` で `Author identity unknown` | user未設定 | S1の `git config --global user.name/email` を入力 |
| 時間が足りない | 日程が厳しい | S9のfallbackで動かすことを優先。`PROTOTYPE_EVAL_PLAN.md` は「動いた範囲 + 残りの計画」で提出可(未実施は未実施と書く) |
