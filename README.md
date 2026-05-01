# Time Series Forecasting — Beijing Air Pollution

A comparative study of two deep learning architectures for multivariate time series forecasting, applied to hourly air pollution data from Beijing (2010–2014), sourced from [Kaggle](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate). Both models predict the next hour's PM2.5 concentration (`pollution`) given a 24-hour window of meteorological and atmospheric features. The models are built entirely in **TensorFlow/Keras** — no PyTorch.

---


## Dataset

The dataset is the [LSTM Multivariate Pollution dataset](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate) from Kaggle, downloaded via `kagglehub`. It contains hourly measurements from the US Embassy in Beijing.

| Column | Description |
|---|---|
| `date` | Timestamp (hourly) |
| `pollution` | PM2.5 concentration (µg/m³) — **target** |
| `dew` | Dew point (°C) |
| `temp` | Temperature (°C) |
| `press` | Atmospheric pressure (hPa) |
| `wnd_dir` | Wind direction (categorical: NE, NW, SE, cv) |
| `wnd_spd` | Wind speed (m/s) |
| `snow` | Cumulative hours of snow |
| `rain` | Cumulative hours of rain |

---

## Features

After preprocessing, the following 12 features are used as model inputs:

```
dew, temp, press, wnd_spd, snow, rain, year, month,
wnd_NE, wnd_NW, wnd_SE, wnd_cv
```

`wnd_dir` is one-hot encoded into four binary columns (`wnd_NE`, `wnd_NW`, `wnd_SE`, `wnd_cv`). A `Season` column is derived from `month` for EDA only and is not fed to the models.

---

## Pipeline Overview

```
Raw CSV
  └── Date parsing, year/month extraction
  └── One-hot encoding of wind direction
  └── Chronological 70/30 train-validation split
  └── MinMaxScaler fit on train, transform on both splits
  └── Sliding window (window = 24 hours)
        ├── X: shape (n_samples, 24, 12)  — 24 hours of 12 features
        └── y: shape (n_samples, 1)        — pollution at hour t+24
  └── Model 1: Stacked LSTM   → fit → evaluate
  └── Model 2: Transformer    → fit → evaluate
```

**Key design choices:**

- The scaler is fit **only on training data** to avoid leakage.
- The split is **chronological**, not random — correct practice for time series.
- Both models predict one step ahead (the pollution value at the end of the window).

---

## Model 1: Stacked LSTM

A straightforward recurrent baseline using Keras' built-in `LSTM` layer.

```
Input: (batch, 24, 12)
  → LSTM(128, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32)
  → Dropout(0.2)
  → Dense(64, relu)
  → Dense(32, relu)
  → Dense(16, relu)
  → Dense(1)           ← linear output, raw pollution value
```

The first LSTM passes its full sequence of hidden states to the second (`return_sequences=True`). The second LSTM outputs only the final hidden state, which the dense head then maps to a scalar prediction.

Training uses Adam (lr=0.001), MSE loss, and early stopping (patience=10, monitoring validation loss).

---

## Model 2: Custom Transformer Encoder (TensorFlow/Keras)

The second model implements a **Transformer encoder** based on a transformer constructed by (TheGradientPath)[https://github.com/samugit83/TheGradientPath/blob/master/Keras/transformers/time_series_forecast/notebook.ipynb] as custom `tf.keras.layers.Layer` subclasses. 

### MultiHeadSelfAttention

```python
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8): ...
```

This layer implements scaled dot-product multi-head attention. This is what each component does:

**Projections (Q, K, V).** Three separate `Dense` layers linearly project the input into query (`Q`), key (`K`), and value (`V`) matrices, each of shape `(batch, seq_len, embed_dim)`. These projections allow the model to learn different representations for "what am I looking for", "what do I offer", and "what is my content".

**Head splitting.** The `embed_dim`-dimensional representation is split across `num_heads` heads. Each head gets `projection_dim = embed_dim // num_heads` dimensions. The reshape is `(batch, seq_len, num_heads, projection_dim)`, then transposed to `(batch, num_heads, seq_len, projection_dim)` so that attention is computed independently per head.

**Scaled dot-product attention.**

```
score        = Q @ K^T                    # (batch, heads, seq, seq)
scaled_score = score / sqrt(d_k)          # prevents softmax saturation
weights      = softmax(scaled_score)      # attention distribution
output       = weights @ V                # weighted sum of values
```

The scaling by `sqrt(d_k)` (the key dimension) keeps the dot products from growing large, which would push the softmax into regions of near-zero gradient.

**Concatenation.** Each head's output is transposed back to `(batch, seq_len, num_heads, projection_dim)` and reshaped to `(batch, seq_len, embed_dim)`, effectively concatenating the heads. A final `Dense` layer (`combine_heads`) mixes the concatenated representation.

The rationale for multiple heads: each head can attend to different temporal dependencies simultaneously — one head might focus on recent hours, another on periodicity.

---

### PositionalEncoding

```python
class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim): ...
```

Transformers have no recurrence and no convolution, so they have no built-in notion of order. Positional encoding injects order information by adding a deterministic signal to the embeddings.

The encoding uses the sinusoidal scheme from *Attention Is All You Need* (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))
```

where `pos` is the position in the sequence (0 to `max_len-1`) and `i` indexes the embedding dimension. Even dimensions get sine, odd dimensions get cosine. The varying frequencies mean that every position gets a unique encoding, and nearby positions get similar encodings. Crucially, this is **added** to the input, so the shape does not change.

In the code, `np.meshgrid` is used to vectorise the construction of the angle matrix before training begins, and the result is cast to a `tf.float32` constant stored as `self.pos_encoding`. During the forward pass, the relevant slice `[:, :seq_len, :]` is added to the input.

---

### TransformerBlock

```python
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): ...
```

A single Transformer encoder block follows the standard pre-norm or post-norm pattern. Here the **post-norm** variant is used (normalisation after the residual addition):

```
# Sub-layer 1: Multi-Head Attention
attn_out = MultiHeadSelfAttention(inputs)
attn_out = Dropout(attn_out)
out1     = LayerNorm(inputs + attn_out)     # residual connection

# Sub-layer 2: Position-wise Feed-Forward Network (FFN)
ffn_out  = Dense(ff_dim, relu)(out1)
ffn_out  = Dense(embed_dim)(ffn_out)
ffn_out  = Dropout(ffn_out)
out2     = LayerNorm(out1 + ffn_out)        # residual connection
```

**Residual connections** (`inputs + attn_out`, `out1 + ffn_out`) allow gradients to flow directly through the addition path, making deep stacks trainable without vanishing gradients.

**LayerNormalization** normalises each token's representation independently across the embedding dimension (not across the batch). This stabilises activations during training and is preferred over BatchNorm for sequence models.

**The FFN** is a two-layer MLP applied identically and independently to each time step. It expands to `ff_dim` (typically 2–4× `embed_dim`) with ReLU, then projects back down to `embed_dim`. This provides non-linear mixing within each position after the attention has mixed information across positions.

---

### TransformerEncoder

```python
class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1): ...
```

Simply stacks `num_layers` `TransformerBlock` instances sequentially. An initial `Dropout` is applied to the positionally-encoded input before the first block. Each block refines the representation, with later blocks able to build on the multi-scale temporal patterns captured by earlier ones.

---

### Full Model

```python
def build_model_t(time_step, embed_dim=64, num_heads=4, ff_dim=128,
                  num_layers=2, dropout_rate=0.2):
```

The full forward pass:

```
Input: (batch, 24, 12)
  → Dense(embed_dim)              # project 12 features → 64-dim embedding space
  → PositionalEncoding(24, 64)    # add sinusoidal position signal
  → TransformerEncoder(2 blocks)  # attend across 24 time steps, twice
  → x[:, -1, :]                   # take only the last time step's representation
  → Dropout(0.2)
  → Dense(1)                      # predict pollution
```

**Why take only the last time step?** After the encoder, every position's representation has attended to all other positions. The last position (`t = 23`, i.e., the most recent hour) is used as a summary of the full 24-hour window. This is a common choice for encoder-only forecasting; it works because self-attention is not causal here (all positions see all others), so the last token's output already aggregates information from the full sequence.

The model is compiled with Adam and MSE loss, and trained with the same early stopping configuration as the LSTM.

---

## Evaluation

Both models are evaluated on the validation set (the most recent 30% of data, chronologically) using:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average magnitude of prediction error in original units (µg/m³) |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily than MAE |
| **Directional Accuracy** | Fraction of consecutive pairs where the sign of the change is predicted correctly |

Predictions are inverse-transformed from the normalised scale back to µg/m³ before metric computation.

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
matplotlib
seaborn
plotly
kagglehub
visualkeras
```

The notebook was run on **Google Colab** with GPU acceleration.

---

## Usage

1. Open `TimeSeriesForecast_AirPollution.ipynb` in Google Colab (recommended) or a local Jupyter environment with GPU.
2. Run all cells in order. The dataset is downloaded automatically via `kagglehub`.
3. The notebook trains the LSTM first, then the Transformer, and plots forecasts and training curves for both.
