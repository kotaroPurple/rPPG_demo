## rPPG アルゴリズム解説（CHROM / POS）

最終更新: 2025-08-24

本資料は Wang, Stuijk, de Haan (TBME 2016, DOI: 10.1109/TBME.2016.2609282) のモデルに基づく rPPG 推定の理論と、CHROM・POS の実装要点をまとめる。Markdown の数式はインラインを `$...$`、別行を `$$...$$` で表記する。

### 1. 観測モデルと皮膚反射
カメラで観測される、顔の皮膚領域のフレーム平均 RGB ベクトルを $\mathbf{s}(t)\in\mathbb{R}^3$ とする。皮膚の反射は定常な肌トーン成分と拍動に伴う微小変調の和で近似でき、照明強度やシェーディングの時間変動 $I(t)$ を含めて、
$$
\mathbf{s}(t) = I(t)\,\big(\mathbf{s}_0 + p(t)\,\mathbf{s}_1\big) + \mathbf{n}(t)
$$
と表す。ここで、$\mathbf{s}_0$ は定常な肌トーン、$\mathbf{s}_1$ は血液容積脈波に由来する小振幅のクロミナンス変調方向、$p(t)$ はスカラー時系列（PPG）、$\mathbf{n}(t)$ は雑音である。$I(t)$ は露出や姿勢変化による全体スケール変動を表す。

小信号近似（$|p(t)|\ll 1$）と、数秒窓内での移動平均 $\overline{\mathbf{s}}$ を用いた正規化により、
$$
\mathbf{r}(t) = \frac{\mathbf{s}(t)}{\overline{\mathbf{s}}} - \mathbf{1} \;\approx\; \frac{I(t)\,p(t)\,\mathbf{s}_1}{\overline{I}\,\mathbf{s}_0} + \text{高次・雑音}
$$
が得られ、強度スケール $I(t)$ の影響が一部打ち消される。肌トーンの正規化方向 $\mathbf{u} = \mathbf{s}_0/\|\mathbf{s}_0\|$ に直交する平面（クロミナンス平面）
$$
\mathcal{P} = \{\mathbf{x}\in\mathbb{R}^3 \mid \mathbf{u}^\top \mathbf{x} = 0\}
$$
への射影によって、照明スケール変動の残差をさらに抑制できる。

### 2. 前処理（共通）
- ROI（頬/額）のフレーム平均 RGB を取得し、各チャネルを移動平均で正規化：$R_n(t)=R/\overline{R}-1$ など。
- 0.7–4.0 Hz のバンドパスでドリフト/高周波を除去（IIRまたはFIR）。
- 数秒の短窓（例: $L\approx1.6\!\sim\!3.0$ s）に区切り、窓ごとに合成信号を生成。

以降、$\mathbf{r}(t)=[R_n(t),G_n(t),B_n(t)]^\top$ と書く。

### 3. CHROM 法
論文で示される色合成の一例は次の通り：
$$
X(t) = 3R_n(t) - 2G_n(t),\quad
Y(t) = 1.5R_n(t) + G_n(t) - 1.5B_n(t).
$$
窓内で標準偏差比 $\alpha=\sigma(X)/\sigma(Y)$ を計算し、
$$
s_{\text{CHROM}}(t) = X(t) - \alpha\,Y(t)
$$
を rPPG 波形とする。これは肌トーン方向成分を相対的に抑え、拍動由来のクロミナンス変化を強調する線形結合である。

### 4. POS 法（Plane-Orthogonal-to-Skin）
POS は肌トーン直交平面内の二つの基底方向
$$
\mathbf{w}_1 = \begin{bmatrix}0\\1\\-1\end{bmatrix},\quad
\mathbf{w}_2 = \begin{bmatrix}-2\\1\\1\end{bmatrix}
$$
に沿って $\mathbf{r}(t)$ を射影し、窓内の分散比でスケールを調整して合成する：
$$
X(t) = \mathbf{w}_1^\top\mathbf{r}(t)=G_n(t)-B_n(t),\quad
Y(t) = \mathbf{w}_2^\top\mathbf{r}(t)=-2R_n(t)+G_n(t)+B_n(t),
$$
$$
\alpha = \frac{\sigma\big(X\big)}{\sigma\big(Y\big)},\quad
s_{\text{POS}}(t) = X(t) + \alpha\,Y(t).
$$
$\alpha$ は窓ごとに更新する。肌トーン方向（ほぼ $[1,1,1]^\top$ に近い）に対する直交平面で合成するため、照明変動と頭部運動に起因する強度変化を効果的に抑える。

### 5. BPM 推定
窓 $[t_0,t_0+L)$ で得られた合成信号 $s(t)$ から、パワースペクトル $S(f)$ を計算し、心拍帯域 $[f_{\min},f_{\max}]$（例: 0.7–4.0 Hz）内で最強ピーク $\hat f$ を検出、BPM を
$$
\widehat{\text{BPM}} = 60\,\hat f
$$
とする。連続窓で時間系列化し、メディアン/移動平均で平滑化する。位相連続性を考慮したピークトラッキング（前回推定近傍を優先）で外れ値を低減できる。

### 6. 品質指標（例）
心拍帯域ピーク強度とその近傍比を用いた SNR 例：
$$
\text{SNR}_{\text{dB}} = 10\log_{10}\frac{\sum_{f\in\mathcal{H}} |S(f)|^2}{\sum_{f\in\mathcal{N}} |S(f)|^2}
$$
ここで $\mathcal{H}$ はピーク周波数近傍の小区間、$\mathcal{N}$ は帯域内の残差領域。閾値を UI に提示し、信頼度インジケータとして利用する。

### 7. 実装メモ
- ROI は複数パッチ（左右頬＋額）を用意し、各パッチの rPPG をロバスト合成（分散やスキューに基づく重み）するとSNRが向上。
- フィルタは線形位相FIR（遅延一定）かIIR（遅延小）を用途に応じて選択。IIR の初期過渡は捨てる。
- カメラの自動露出/ホワイトバランスは完全固定できない場合があるため、比率正規化と平面射影が重要。
- FPS が低い場合はナイキストに注意し、BPM上限を自動的に引き下げる（例: 上限 < 0.45×FPS）。

### 8. 参考式（実装用の擬似コード）
窓長 $L$（サンプル数）、信号 $R,G,B$：
1. 正規化：$R_n=R/\overline{R}-1$（移動平均）、同様に $G_n,B_n$
2. バンドパス：$R_n,G_n,B_n\leftarrow$ filter(0.7–4.0 Hz)
3. CHROM：$X=3R_n-2G_n$, $Y=1.5R_n+G_n-1.5B_n$, $\alpha=\sigma(X)/\sigma(Y)$, $s=X-\alpha Y$
4. POS：$X=G_n-B_n$, $Y=-2R_n+G_n+B_n$, $\alpha=\sigma(X)/\sigma(Y)$, $s=X+\alpha Y$
5. BPM：FFTピークから $\widehat{\text{BPM}}$

### 9. 参考文献
- Wang, W., den Brinker, A. C., Stuijk, S., de Haan, G. Algorithmic Principles of Remote Photoplethysmography. IEEE Trans. Biomed. Eng., 2017. DOI: 10.1109/TBME.2016.2609282
