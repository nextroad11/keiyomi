# 競読みAI（keiyomi）

JRA公式サイトの出馬表・オッズ・結果を自動で集め、AIの確率から買い目を出す競馬予想アプリ。
しくみは艇読みAI（teiyomi）と同じ。

| 役割 | 担当 | 場所 |
|---|---|---|
| データ取得 | GitHub Actions | `.github/workflows/`, `scraper/` |
| 予想の計算 | GitHub Actions（開催日に1分ごと） | `model/predict.py`、学習は `model/train.py`（週1回） |
| 表示 | ブラウザ（GitHub Pages、合言葉で開く） | `docs/index.html` |
| データ置き場 | `data` ブランチ | 暗号化したJSONのみ |

## 決まりごと
- JRA公式サイトへのアクセスは1秒に1回まで。
- 取ったデータは生データも含めてすべて暗号化してから置く（合言葉 → PBKDF2-SHA256 31万回 → AES-256-GCM）。合言葉は GitHub Secrets の `APP_PASS` だけに置く。
- 取得できない項目は推測で埋めず「取得不可」にする。

## いまの段階
1. 下見（`probe.yml`）：済み。GitHub Actions から JRA に届き、出馬表・オッズ（単複〜3連単）・結果・払戻のページを読めることを確認した。
2. 過去データの取り込み（`history.yml`、`scraper/history.py`）：今月から60か月（5年）さかのぼり、各レースの結果と確定の単勝・複勝オッズを取る。1時間おきに40分ずつ、続きから。
   - 保存先：`data` ブランチの `enc/results/YYYYMMDD.json`（合言葉で暗号化）
   - 開発用の控え：`dev/`（`keys/dev_pub.txt` の公開鍵で封。秘密鍵は Claude の作業環境だけ）

3. 過去の確定3連複・3連単オッズ（`odds-history.yml`、`scraper/odds_history.py`）：過去結果の各レースについて取り込む。置き場は `odds` ブランチ。
4. 開催日の取得（`daily.yml`、`scraper/daily.py`）：開催日の8〜18時に1分ごと。出馬表、発走30・20・15・10・5・3・1分前の単複オッズ、30・10・5・3・1分前の3連複・3連単オッズ、結果と確定オッズ。遅れて取れなかった時点は空けておく。

5. 本番用の学習（`train.yml`、`model/train.py`）：週1回、使えるデータ全部で学習し直して `enc/model.json`。
6. 開催日の予想と記録（`model/predict.py`、`daily.yml` の中で1分ごと）：最新オッズで AI 確率と3連複・3連単の買い目。締切5分前のオッズで買い目を固定し、結果が出たら実際の払戻で採点（まだ賭けない）。`enc/pred/`、`enc/paper/records.json`。

## 検証
- `model/REPORT.md` に結果。未使用期間の数字だけを報告する。

## JRAのページの読み方（`scraper/jra.py`）
- ページは `cname` を POST して開く。cname 末尾の `/XX` はサイトが付ける検査文字なので自作しない。必ずリンクからたどる。
- cname の最後の20桁＝場コード2・年4・回2・日2・R2・日付8。レースIDは `日付-場-R`（例 `20261114-05-01`）。
- 場コード：01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉
