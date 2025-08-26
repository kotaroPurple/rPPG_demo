## 開発タスク一覧

目的: uv + DearPyGUI を用いたデスクトップrPPGアプリを実装し、将来のWeb UI連携を想定したサービス構成を整える。進捗に応じてチェックを更新する。

### ドキュメント/設計
- [x] 要求仕様書を作成（`docs/01_requirements.md`）
- [x] rPPGアルゴリズム解説（数式$/$表記統一）（`docs/02_rppg_algorithm.md`）
- [x] コントリビュータ向けガイド（`AGENTS.md`）
- [x] 設計書（デスクトップ＋将来Web対応）（`docs/03_design.md`）

### 環境/依存
- [x] 依存の追加（`uv add opencv-python dearpygui mediapipe numpy scipy`）
- [x] フォーマッタ/リンタ導入（`ruff`）設定を `pyproject.toml` に追加
- [x] `ruff` インストール（`uv add ruff`）と実行手順整備
- [x] taskipy 導入・タスク定義（`uv run task run|lint|fmt`）

### 実装（Desktop/Core）
- [x] パッケージ骨組み `src/rppg/` を作成（`__init__.py`, `app.py` スタブ）
- [x] DearPyGUI 最小プレビューとBPM表示を実装（カメラ→平均RGB→POS→BPM）
- [x] `capture.py`: カメラ取得（OpenCV）、FPS計測、タイムスタンプ（UI組込み）
- [x] `roi.py`: 顔検出（MediaPipe）＋スキンマスク（頬/額マスク）
- [x] `preprocess.py`: 正規化（移動平均）＋バンドパス（0.7–4.0 Hz）
- [x] `chrom.py`/`pos.py`: CHROM/POS 合成
- [x] `bpm.py`: FFTピーク検出、BPMレンジ制約
- [x] スペクトル表示とSNR表示を追加
- [x] プレビューに顔矩形オーバーレイ（認識可視化）
- [x] GUI: 推定BPMの時系列プロットを追加（移動窓で更新）
- [x] GUI: rPPG波形（BPM算出前）のプロットを整備
- [x] `quality.py`: SNR/ピーク信頼度算出
- [x] `recorder.py`: CSV/JSON 非同期保存
- [x] `app.py`（DearPyGUI）: プレビュー、波形/スペクトル、BPM/品質、設定

### Web連携（将来・現時点では非実施）
- [ ] `service.py`: FastAPI + WebSocket でメトリクス配信
- [ ] Webフロント雛形 `web/frontend/`（静的UI or SPA）

### テスト/検証
- [ ] `pytest` 導入と基本テスト（前処理/合成/BPM）
- [ ] カメラ依存をモック化したユニットテスト
- [ ] 最低カバレッジ閾値（例: 80%）の設定

### 運用/品質
- [ ] 実行スクリプト追加例: `uv run python -m rppg.app`
 - [ ] README 更新（起動手順、注意点、既知の制約）
- [ ] 動作確認（屋内拡散照明、軽微動作で HR MAE ≤ 5 BPM）

### デバッグ/安定化（追加）
- [x] ロギングとfaulthandler追加（`logs/`）
- [x] Spectrum描画の間引き/点数制限・Barsモード追加
- [x] SpectrumデフォルトをOFF
 - [x] 既知のトラブルシュートをREADMEに記載（Continuity Camera, Spectrum描画）

### Web連携（将来）
- [x] FastAPI サービスへ現在値メトリクスを提供（`/metrics`, `/ws`）
- [ ] Web UI 雛形（メトリクスの簡易表示）

更新ポリシー: タスク完了時にこのファイルにチェックを入れ、コミットする。
