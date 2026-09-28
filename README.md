# Information Theory for Crypto Market Data

Coursework project applying information-theoretic tools (Shannon entropy,
mutual information) to 30-minute Binance futures data for the largest USDT
contracts (BTC, ETH, SOL and others).

## Contents

| File | Description |
|------|-------------|
| `Information_Theory.ipynb` | Theory notes with numerical examples: entropy, joint and conditional entropy, KL divergence, mutual information, conditional mutual information |
| `First set of experiments.ipynb` | Rolling Shannon entropy of price, buy volume, sell volume and volume difference (BTC, ETH, SOL); Ridge and logistic-regression experiments testing whether entropy features explain the size of price moves |
| `Second set of experiments.ipynb` | Pairwise and triple mutual information between assets (log returns, buy/sell volumes); per-asset MI feature matrix and hierarchical clustering of assets by MI profile |
| `load_features.py` | Builds 30-minute features (last trade price, buy and sell volume) from raw trade files |
| `join_features.py` | Merges per-instrument features into one time-by-asset panel |

## Method

1. Resample trades to 30-minute bars: close price, buy volume, sell volume.
2. Within a rolling window (1 day = 48 bars, or 1 week = 336 bars),
   discretize each series into 5 quantile bins.
3. Compute Shannon entropy of the bin distribution in each window.
   Low entropy means the window is dominated by a few states, high entropy
   means values are spread across bins.
4. Regress the absolute and signed price move over the last 48 bars, and
   classify whether the move exceeds a threshold, using entropy features
   (price only, or price + buy/sell volume). Models: Ridge, logistic regression.
5. Estimate pairwise and triple mutual information between assets, build
   per-asset MI statistics (mean, std, min, max, median) and cluster assets
   with hierarchical clustering.

## Results

Exploratory. Data: August-September 2024, training on Aug 1 - Sep 14,
validation on the following days.

- Entropy features show some association with the size of price moves on
  the daily window (validation R² of about 0.2-0.6 for some asset/feature
  combinations, e.g. ETH 0.63 with price entropy only).
- The picture is not stable: results for signed returns are mostly negative
  in R², weekly-window models are mostly negative, and several classifiers
  degenerate to predicting a single class.
- Conclusion: entropy features carry a weak, inconsistent signal on this
  sample; no claim of predictive power is made.

## Limitations

- One month of training data, three assets, validation windows of 2-7 days.
- The target (price move over the last 48 bars) overlaps with the window
  used to compute the entropy, so the experiments measure contemporaneous
  association rather than out-of-sample forecasting.
- No transaction-cost or trading simulation.

## Data and reproducibility

Input: Binance USDT-margined futures trades (timestamp, side, price,
quantity), aggregated to 30-minute bars. The raw data is not included and the
loading scripts depend on a private research environment, so the notebooks
are provided for reading, not for re-running end to end.
