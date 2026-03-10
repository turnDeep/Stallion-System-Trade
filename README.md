# Stallion-System-Trade 🐎📈

**Stallion-System-Trade** は、米国株ラッセル3000銘柄を対象とした、完全自律型の「**損小利大・1日1回限定・オープニングレンジブレイクアウト（ORB）**」専用デイトレードBotシステムです。

Webull証券（米国株・現物取引）のV2 Trade APIで注文を執行し、FinancialModelingPrep（FMP）APIでリアルタイム価格を取得します。週末の機械学習ベースの銘柄スクリーニングから、平日のリアルタイム自動発注まで、Dockerコンテナ上で24時間365日**全自動**で稼働します。

実績として、過去のホールドアウト（OOS）検証において、取引手数料（片道0.2%）を完全に加味した上で、**Sharpe比5.07・最大ドローダウン-3.97%・ネット+20%超**のパフォーマンスを確認しています。

---

## 🌟 システムの特徴（The "Stallion" Philosophy）

このシステムは「勝率や取引回数」を意図的に捨て、**「1回の爆発力」**に全振りした尖ったハイリスク・ハイリターン哲学を持っています。

1. **先行指標（Ex-Ante）スコアリングによる「最強の暴れ馬トップ10」の抽出:**
   毎営業日の引け後、過去40日分の5分足データから「損益に依存しない先行指標」を計算し、Z-Scoreで銘柄をランキングします。
   使用する先行指標：**ADR（日中値幅率）・ATR%（真の値幅率）・ADV（平均出来高）・AvgGap（平均ギャップ率）**
   ※ PCAや過去の損益実績には依存しないため、将来のパフォーマンス予測として過適合（overfitting）しにくい設計です。

2. **1日1回限定・資金90%の「一点集中フルインベストメント」:**
   取引回数が多いほど手数料負けするデイトレードの罠を回避するため、選抜されたトップ10銘柄の中で**「その日一番最初にブレイクアウトした1銘柄だけ」**に、総資金の90%を集中投資します。
   エントリー窓は **09:35〜10:30 (EST)** に限定し、それを過ぎると当日はトレードなし。

3. **3段階ダイナミックエグジット（Dynamic Exit）による優れたリスク管理:**
   固定利確・固定損切りを廃止し、以下の3-phase構造でエグジットを管理します。
   - **フェーズ1 ブレイクイーブン移動:** 含み益が初期リスク（R）×0.75に達したらストップをエントリー価格に移動（元本保護）
   - **フェーズ2 トレーリングストップ:** 含み益がR×1.5に達したらトレーリング開始（ストップ = 最高値 − R×0.75）
   - **フェーズ3 EOD強制決済:** 15:55 (EST) に全株を成行売り（オーバーナイト・持ち越しリスクゼロ）

4. **FMP APIによるリアルタイム価格取得:**
   yfinance等の遅延APIを排除し、FinancialModelingPrep（FMP）の `/quote` エンドポイントを2秒間隔でポーリングします。

---

## 📦 アーキテクチャと稼働スケジュール

```
master_scheduler.py
├── 毎営業日 17:00 (EST) → ml_pipeline_60d.py を実行
│     └── Ex-Ante Z-Scoreで Top 10 銘柄を選定 → top_10_watchlist.json に保存
└── 毎営業日 09:25 (EST) → webull_live_trader.py を実行
      ├── 09:30〜09:35: 最初の5分間（Opening Range）の高値をFMPで取得
      ├── 09:35〜10:30: ブレイクアウト監視 → 1回だけ成行買い注文（Webull V2 API）
      ├── エントリー後: DynamicExitStrategy（strategy.py）でティック単位の出口管理
      └── 15:55: 未決のポジションを強制全決済
```

**モジュール構成：**

| ファイル | 役割 |
|---|---|
| `config.py` | FMP / Webull 認証情報の一元管理 |
| `strategy.py` | `DynamicExitStrategy`・`check_entry_condition()`（共通モジュール） |
| `ml_pipeline_60d.py` | Ex-Ante Z-Scoreウォッチリスト生成（daily, 引け後実行） |
| `webull_live_trader.py` | リアルタイム執行Bot（FMP quotes + Webull注文） |
| `backtester.py` | `strategy.py` と同一ロジックを使用した歴史的バックテスト |
| `master_scheduler.py` | 全スクリプトのスケジューリング司令塔 |

---

## 🚀 構築手順（Deploy via Docker）

### 1. リポジトリのクローン
```bash
git clone https://github.com/turnDeep/Stallion-System-Trade.git
cd Stallion-System-Trade
```

### 2. 環境変数（API認証情報）のセット
プロジェクトのルートディレクトリに `.env` ファイルを作成します。
```env
WEBULL_APP_KEY="あなたのWebull App Key"
WEBULL_APP_SECRET="あなたのWebull App Secret"
WEBULL_ACCOUNT_ID="あなたのWebull 口座ID"
FMP_API_KEY="あなたのFinancialModelingPrep API Key"
```

### 3. Docker Composeで一撃デプロイ
```bash
docker compose up -d
```
> ※ `Dockerfile` にてコンテナの内部時刻を米国（America/New_York）に強制指定しているため、サーバー側の時刻を気にする必要はありません。

---

## 📊 検証済みパフォーマンス（OOS Holdout）

| 指標 | 新Baseline（固定エグジット） | 新Baseline + Dynamic Exit |
|---|---|---|
| 総リターン | +27.74% | +20.47% |
| Sharpe比 | 5.13 | **5.07** |
| 最大ドローダウン | -11.56% | **-3.97%** |
| 手数料（片道） | 0.2% | 0.2% |

> Dynamic Exitは絶対リターンをわずかに犠牲にしつつ、最大ドローダウンを大幅に圧縮します。実運用目線では**Dynamic Exit**を採用しています。

---

## 🛠 必須要件（Requirements）

* Docker および Docker Compose がインストールされたLinux環境
* Webull証券（日本）の口座、および OpenAPI（V2 Trade）の利用権限
* FinancialModelingPrep（FMP）のAPIキー（無料プランでも可）
* ラッセル3000の5分足データ（`russell3000_60d_5min.pkl`）

---
*Developed as an automated high-volatility ORB breakout hunting system.*
