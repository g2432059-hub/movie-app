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
