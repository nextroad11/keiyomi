# 競読みAI（keiyomi）

JRA公式サイトの出馬表・オッズ・結果を自動で集め、AIの確率から買い目を出す競馬予想アプリ。
しくみは艇読みAI（teiyomi）と同じ。

| 役割 | 担当 | 場所 |
|---|---|---|
| データ取得 | GitHub Actions | `.github/workflows/`, `scraper/` |
| 予想の計算と表示 | ブラウザ（GitHub Pages） | `docs/`（準備中） |
| データ置き場 | `data` ブランチ | 暗号化したJSONのみ |

## 決まりごと
- JRA公式サイトへのアクセスは1秒に1回まで。
- 取ったデータは生データも含めてすべて暗号化してから置く（合言葉 → PBKDF2-SHA256 31万回 → AES-256-GCM）。合言葉は GitHub Secrets の `APP_PASS` だけに置く。
- 取得できない項目は推測で埋めず「取得不可」にする。

## いまの段階
1. 下見（`probe.yml`）：GitHub Actions から JRA に届くか、ページの作りを確かめる。取ったページは `keys/probe_pub.txt` の公開鍵で封をして `data` ブランチの `debug/probe/` に置く。
