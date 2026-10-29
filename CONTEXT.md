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

- `python evaluate.py splits/test` は、人間だけが実行する。Copilot は実行しない。`yes |` のようなパイプで確認を自動で通さない。
