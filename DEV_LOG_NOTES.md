# 開発記録のメモ(DEV_LOG.md を書くときの材料)

## 2026-10-24 S5 make_split.py
- 依頼: 利用者単位でtrain/dev/testに分け、dev/testの入力5本と評価用を作るscriptをCopilotに依頼
- 問題: Copilotが「test_inputs.jsonを作るには、make_split.pyがtest利用者の評価を読む必要がある。CONTEXT.mdとプロンプトの例外の書き方が食い違っている」と質問した
- 確認: test の中身は表示せず、人数と評価件数だけを出す条件で許可。sha256sumで2回の実行結果が一致、重なり0を確認
- 修正: CONTEXT.mdとマニュアルの例外の書き方を、make_split.pyとevaluate.pyの両方に揃えた

## 2026-10-24 S7-S8 学習結果(val: train利用者の評価の5%)
- mean 0.9772 / MF 0.8556(7epoch) / NCF 0.8783(9epoch)。NCFはMFより約0.023悪い(調整は最小限)
- 理由の候補: MLPが内積に劣る場合がある既知の報告(要確認)、NCFに偏りの項がない、調整不足

## 2026-10-24 S10 evaluate.py 全dev実行で失敗
- 依頼: dev/testを評価するevaluate.pyをCopilotに依頼。--limit 50 では全モデル正常に動いた
- 問題: 全dev(604人)の --all で ValueError(映画ID 133 は対応表にない)。dev利用者の入力5本に、train利用者の評価がなく対応表にない映画が含まれていた
- 確認: 全dev実行でエラーを再現。validate_inputs はアプリ用に対応表外を拒否する仕様
- 修正: 評価側で対応表外の入力映画を除いて fold-in し、3本未満の利用者は全モデルで対象外にした(recommender.py は変更せず)。結果は延べ2件(2人)が影響を受け、除外0人

## 2026-10-24 順位づけがクリップ後の予測値だった不具合
- 発見: アプリで高評価ばかりを入力したところ、予測評価が 5.00 で同点になった(アニメ5本の入力で、生予測 5.0 以上が67本)
- 原因: recommend() と evaluate.py が、1〜5にクリップした値で順位づけをしていた。同点は movie_id 昇順で並ぶため、順位が失われていた
- 修正: 順位づけにはクリップ前の値を使い、クリップは表示とRMSEの計算だけに使う。テストを2件追加(計23件が通る)
- 結果(dev 604人): RMSEとmean/popular/randomは不変。MF/NCFのPrecision@10 top25は +0.0015〜+0.0091 わずかに上昇(MF 0.2035→0.2126、NCF λ=0 0.1735→0.1781)。人気順(0.296)との差は縮まらなかった
- 旧結果は results/dev_all_default.txt、新結果は results/dev_all_rankfix.txt に保存。testは使っていない

## 2026-10-30 test 1回目(改善前)を実行
- 実行: 2026-10-30 00:48、commit 08dec75(既定λ=0.25を固定した後)、`python -u evaluate.py splits/test --all`。確認のプロンプトには手入力で yes と答えた
- 経緯: 23:44に最初の起動をしたが、確認のプロンプトで止まり、評価は始まっていない(その時点で results/test_runs.log なし、結果のCSVなし)。Copilot CLI が yes を起動して重く動いていたため、終了してから実行した
- 結果: devと同じ傾向。λ=0.25でP@10は λ=0 比 −15.2%、ジャンル数 9.32→13.91。人気順が最高(0.306)、RMSEは平均評価が最良(0.986)
- testの結果を見て設定は変えていない。test残り1回(改善後に使う)
