## rPPG デスクトップアプリ 設計

最終更新: 2025-08-24

本設計は uv + DearPyGUI を前提に、内蔵カメラから rPPG（CHROM/POS）を推定して BPM を表示・保存する最小〜拡張アーキテクチャを示す。

## 全体アーキテクチャ
```mermaid
flowchart TD
  CAM[Camera (OpenCV)] --> CAP[Capture Layer]
  CAP -->|Frame (BGR, ts)| ROI[ROI/Face Tracking]
  ROI -->|ROI mean RGB, ts| CORE[rPPG Core\n(Preprocess, CHROM/POS, Filter)]
  CORE -->|signal window, BPM, SNR| UI[DearPyGUI UI]
  CORE --> REC[Recorder\n(CSV/JSON)]
  subgraph Services
    CAP
    ROI
    CORE
    REC
  end
```

- Capture: カメラからフレーム＋タイムスタンプ取得、FPS計測
- ROI: 顔検出/ランドマーク、スキンマスク生成、複数パッチの追跡
- Core: 正規化・バンドパス・CHROM/POS合成・BPM推定・品質指標
- UI: プレビュー（ROI重畳）、波形/スペクトル、BPM/品質、設定
- Recorder: R/G/B 平均、rPPG、BPM を非同期保存

## スレッド/非同期モデル
```mermaid
sequenceDiagram
  participant Cam as Capture Thread
  participant Proc as Proc Worker
  participant UI as UI Main (DearPyGUI)
  Cam->>Proc: Frame(ts)
  Proc->>Proc: ROI抽出→前処理→CHROM/POS→BPM
  Proc-->>UI: UIイベント(波形更新, BPM, SNR)
  Proc-->>Recorder: サンプル追記
  UI->>Cam: 設定変更(解像度/FPS/ROI)
```
- UIスレッドは描画と操作に限定。処理はWorkerで実行（`queue.Queue` で受け渡し）。
- 設定変更はスレッドセーフな共有構造（`dataclasses` + `threading.Lock`）で反映。

## モジュール構成（`src/` 提案）
```
src/
  rppg/
    __init__.py
    capture.py        # OpenCV カメラ制御, FPS, 色空間変換
    roi.py            # MediaPipe顔検出/ランドマーク, スキンマスク, パッチ定義
    preprocess.py     # 正規化(移動平均), バンドパス(FIR/IIR)
    chrom.py          # CHROM 合成
    pos.py            # POS 合成
    bpm.py            # スペクトル/ピークトラッキング, BPM計算
    quality.py        # SNR等の品質指標
    recorder.py       # CSV/JSON 非同期書き込み
    app.py            # DearPyGUI エントリ, ループ, 配線
```

## データモデル
- Frame: `np.ndarray(BGR)`, `timestamp: float`
- RoiResult: `mask(s)`, `valid: bool`, `landmarks`
- Sample: `mean_rgb: (R,G,B)`, `timestamp`
- WindowResult: `signal: np.ndarray`, `bpm: float`, `snr: float`, `peak_f: float`

## 信号処理設計
- 正規化: $x_n(t) = x(t)/\overline{x}(t) - 1$（窓内移動平均）
- バンドパス: 0.7–4.0 Hz（IIR: Butterworth 2–4次／FIR: 0相）
- 合成:
  - CHROM: $X=3R_n-2G_n,\ Y=1.5R_n+G_n-1.5B_n,\ s=X-\alpha Y,\ \alpha=\sigma(X)/\sigma(Y)$
  - POS: $X=G_n-B_n,\ Y=-2R_n+G_n+B_n,\ s=X+\alpha Y$
- BPM: 心拍帯域でFFTピーク→連続窓で平滑化（メディアン/EMA）

## UI レイアウト（DearPyGUI）
- 左: カメラプレビュー（ROI枠/マスク重畳）
- 右上: BPM, SNR, FPS
- 右中: 波形プロット、下: スペクトル
- 右下: 設定パネル（アルゴリズム、窓長/ステップ、帯域、BPM範囲、記録）

## 主要パラメータ（初期値）
- 窓長/ステップ: 2.0s / 1.0s（50%重なり）
- 帯域: 0.7–4.0 Hz、BPM範囲: 42–240
- ROI: 頬×2＋額、重み=面積 or 分散逆数

## エラーハンドリング
- カメラ未接続/権限: UI通知＋再試行
- 顔未検出: 前回ROI保持→タイムアウトで停止/再検出
- FPS不足: 自動ダウンサンプリング、BPM上限を `0.45×FPS` に制限

## ログ/保存
- CSV: `timestamp,R,G,B,signal,bpm,snr`
- JSONメタ: 設定、開始/終了時刻、デバイス情報

## 将来拡張
- マルチパッチのロバスト合成（RANSAC/重み最適化）
- 動き補償（光フロー）/自動WB固定
- 品質に基づく適応窓長
