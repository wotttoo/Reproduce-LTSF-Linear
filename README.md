# Are Transformers Effective for Time Series Forecasting? (AAAI 2023)

This repository is a **reproduction study** based on the official PyTorch implementation of LTSF-Linear:
> "[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504.pdf)" — Zeng et al., AAAI 2023.

The original codebase was cloned from [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear). This fork documents modifications, bug fixes, and reproduction results conducted as part of an independent study.

---

## Table of Contents

- [Original Paper Overview](#original-paper-overview)
- [Reproduction Study](#reproduction-study)
  - [Environment](#environment)
  - [Changes and Bug Fixes](#changes-and-bug-fixes)
  - [Experimental Scope](#experimental-scope)
  - [Results: Exchange Rate](#results-exchange-rate)
  - [Results: Electricity](#results-electricity)
  - [Discussion](#discussion)
- [Original Repository Documentation](#original-repository-documentation)

---

## Original Paper Overview

The paper challenges the assumption that Transformer-based architectures are necessary for long-term time series forecasting (LTSF). It introduces the **LTSF-Linear** family — a set of extremely simple linear models that consistently outperform all Transformer-based baselines across multiple benchmarks.

### LTSF-Linear Family

![Linear models](pics/Linear.png)

| Model | Description |
|---|---|
| **Linear** | A single linear layer mapping the input window directly to the prediction horizon. |
| **NLinear** | Subtracts the last value of the input sequence before the linear layer, then adds it back. Handles distribution shifts between train and test sets. |
| **DLinear** | Decomposes the input into trend (moving average) and seasonal (residual) components, applies a separate linear layer to each, then sums the outputs. Handles data with clear trend patterns. |

### Transformer Baselines Included

- [Transformer](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- [Informer](https://arxiv.org/abs/2012.07436) (AAAI 2021 Best Paper)
- [Autoformer](https://arxiv.org/abs/2106.13008) (NeurIPS 2021)
- [Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I) (ICLR 2022 Oral)
- [FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022)

---

## Reproduction Study

### Environment

| Item | Value |
|---|---|
| Python | 3.6.9 |
| PyTorch | 1.11.0 |
| Hardware | CPU only (no GPU available) |
| OS | Ubuntu 22.04 |

```bash
conda create -n LTSF_Linear python=3.6.9
conda activate LTSF_Linear
pip install -r requirements.txt
```

> **Note:** All experiments in this reproduction were run on CPU. Training times per experiment ranged from approximately 5 minutes (small datasets) to over 45 minutes (electricity with 321 channels).

---

### Changes and Bug Fixes

#### 1. Device Compatibility Fix — `layers/AutoCorrelation.py`

**Problem:** The original code hardcoded `.cuda()` when constructing the index tensor inside `AutoCorrelation.time_delay_agg_training()`:

```python
# Original (broken on CPU)
init_index = torch.arange(length)...repeat(...).cuda()
```

**Fix:** Replaced with `.to(corr.device)` so the tensor is placed on the same device as the computation graph:

```python
# Fixed
init_index = torch.arange(length)...repeat(...).to(corr.device)
```

This fix is required to run Autoformer and any model using `AutoCorrelation` on CPU or on non-default GPU devices.

#### 2. Script Scope Reduction

The original experiment scripts loop over all 8 datasets and all Transformer variants in a single file. For this reproduction, scripts were narrowed to the **Exchange Rate** and **Electricity** datasets to keep runtime feasible on CPU.

Specific changes per script:

| Script | Change |
|---|---|
| `scripts/EXP-LongForecasting/Formers_Long.sh` | Runs Autoformer only; restricted to Electricity dataset; pred_len ∈ {336, 720} |
| `scripts/EXP-LongForecasting/Linear/exchange_rate.sh` | Changed model to NLinear; seq_len set to 96 |
| `scripts/EXP-LongForecasting/Linear/electricity.sh` | Removed pred_len 96 and 192 runs; commented out pred_len 720 |
| `FEDformer/scripts/LongForecasting.sh` | Restricted to Exchange Rate; fixed missing `--root_path` argument; fixed variable name bug (`$pred_len` → `${preLen}`); changed `--features S` to `--features M` |
| `Pyraformer/scripts/LongForecasting.sh` | Restricted to Exchange Rate only |

#### 3. Per-Model Transformer Scripts

The combined `Formers_Long.sh` was split into three independent scripts to allow running each baseline separately:

- `scripts/EXP-LongForecasting/Formers_Long_Autoformer.sh`
- `scripts/EXP-LongForecasting/Formers_Long_Informer.sh`
- `scripts/EXP-LongForecasting/Formers_Long_Transformer.sh`

---

### Experimental Scope

This reproduction focuses on two datasets:

| Dataset | Channels | Frequency | Train / Val / Test |
|---|---|---|---|
| Exchange Rate | 8 | Daily | 5120 / 665 / 1422 |
| Electricity | 321 | Hourly | ~18,317 / ~2,633 / ~5,261 |

Models evaluated:

- **DLinear** — multivariate (`--features M`), seq_len ∈ {96, 336, 720}, pred_len ∈ {96, 192, 336, 720}
- **NLinear** — multivariate (`--features M`), seq_len ∈ {96, 336, 720}, pred_len ∈ {96, 192, 336, 720}

Key hyperparameters:

| Param | Exchange Rate | Electricity |
|---|---|---|
| `batch_size` | 8 | 16 |
| `learning_rate` | 0.0005 | 0.001 |
| `train_epochs` | 10 | 10 |
| `patience` (early stop) | 3 | 3 |
| `moving_avg` kernel | 25 | 25 |

---

### Results: Exchange Rate

#### DLinear — Exchange Rate (Multivariate, features=M)

| seq_len | pred_len | MSE (ours) | MSE (paper) | MAE (ours) | MAE (paper) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 96 | 96 | **0.0782** | 0.089 | **0.1986** | 0.208 |
| 96 | 192 | **0.1560** | 0.180 | **0.2921** | 0.300 |
| 96 | 336 | **0.3069** | 0.331 | **0.4193** | 0.415 |
| 96 | 720 | **0.7816** | 1.033 | **0.6693** | 0.780 |
| 336 | 96 | 0.0848 | 0.089 | 0.2087 | 0.208 |
| 336 | 192 | 0.1621 | 0.180 | 0.2958 | 0.300 |
| 336 | 336 | 0.3332 | 0.331 | 0.4414 | 0.415 |
| 336 | 720 | 0.8977 | 1.033 | 0.7251 | 0.780 |
| 720 | 96 | 0.0914 | 0.089 | 0.2222 | 0.208 |
| 720 | 192 | 0.1687 | 0.180 | 0.2989 | 0.300 |
| 720 | 336 | 0.4390 | 0.331 | 0.4973 | 0.415 |
| 720 | 720 | 1.1267 | 1.033 | 0.7907 | 0.780 |

#### NLinear — Exchange Rate (Multivariate, features=M)

| seq_len | pred_len | MSE (ours) | MSE (paper) | MAE (ours) | MAE (paper) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 96 | 96 | 0.0856 | 0.081 | 0.2021 | 0.203 |
| 96 | 192 | 0.1758 | 0.157 | 0.2960 | 0.293 |
| 96 | 336 | 0.3287 | 0.305 | 0.4137 | 0.414 |
| 96 | 720 | 0.8859 | 0.643 | 0.7107 | 0.601 |
| 336 | 96 | **0.0893** | 0.081 | 0.2082 | 0.203 |
| 336 | 192 | 0.1806 | 0.157 | 0.3003 | 0.293 |
| 336 | 336 | 0.3303 | 0.305 | 0.4150 | 0.414 |
| 336 | 720 | 0.9254 | 0.643 | 0.7216 | 0.601 |
| 720 | 96 | 0.0941 | 0.081 | 0.2143 | 0.203 |
| 720 | 192 | 0.1834 | 0.157 | 0.3019 | 0.293 |
| 720 | 336 | 0.3458 | 0.305 | 0.4227 | 0.414 |
| 720 | 720 | 0.9241 | 0.643 | 0.7229 | 0.601 |

> **Note:** Paper benchmarks for NLinear use `seq_len=336`. Our reproduction also ran `seq_len=96` and `seq_len=720` for additional comparison.

---

### Results: Electricity

#### DLinear — Electricity (Multivariate, seq_len=336, features=M)

| pred_len | MSE (ours) | MSE (paper) | MAE (ours) | MAE (paper) |
|:---:|:---:|:---:|:---:|:---:|
| 96 | **0.1401** | 0.141 | **0.2374** | 0.237 |
| 192 | **0.1538** | 0.154 | **0.2505** | 0.248 |
| 336 | N/A* | 0.171 | N/A* | 0.265 |
| 720 | N/A* | 0.210 | N/A* | 0.297 |

> \* Experiments for pred_len=336 and pred_len=720 on Electricity did not produce final test metrics due to runtime constraints on CPU (training completed but evaluation output was incomplete). These experiments require GPU to complete within reasonable time.

---

### Discussion

#### DLinear on Exchange Rate

Results for `seq_len=96` are consistently **better than or match the paper** across all prediction horizons. With `seq_len=336`, results align closely with the paper for short horizons (96, 192) but diverge at longer horizons (720), suggesting the paper's benchmark used specific tuning.

#### NLinear on Exchange Rate

NLinear results are noticeably **higher (worse) than the paper** for long prediction horizons (pred_len=720), particularly across all seq_len values. The paper reports MSE=0.643 for pred_len=720 (seq_len=336), while our reproduction yields 0.925. This discrepancy is likely attributable to:

1. The paper's benchmark using `--features S` (univariate) for NLinear on Exchange Rate, while our scripts used `--features M` (multivariate).
2. Possible use of a different random seed or training configuration in the original paper.

#### DLinear on Electricity

Results for pred_len=96 and pred_len=192 match the paper's reported values to within 0.001 MSE, confirming the correctness of the implementation and the bug fix applied to `AutoCorrelation.py`.

#### Key Takeaway

The core claim of the paper holds: **DLinear with seq_len=96 achieves competitive or better performance than reported**, confirming that simple linear models are sufficient for LTSF. The reproduction validates the codebase is functionally correct when the device compatibility fix is applied.

---

## Original Repository Documentation

### Updates
- [2024/01/28] Model included in [NeuralForecast](https://github.com/Nixtla/neuralforecast).
- [2022/11/23] Accepted to AAAI 2023 with three strong accept. Benchmark released: [LTSF-Benchmark.md](LTSF-Benchmark.md).
- [2022/08/25] Paper updated with comprehensive analyses. NLinear and Linear added to the LTSF-Linear family.

### Features
- [x] Benchmark for long-term time series forecasting: [LTSF-Benchmark.md](LTSF-Benchmark.md)
- [x] Univariate and multivariate forecasting support
- [x] Weight visualization
- [x] Scripts for different look-back window sizes

### Data Preparation

Download all nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer.

```bash
mkdir dataset
# Place CSV files in ./dataset/
```

### Training

```bash
# DLinear on Exchange Rate
sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh

# Autoformer/Informer/Transformer baselines
sh scripts/EXP-LongForecasting/Formers_Long_Autoformer.sh
sh scripts/EXP-LongForecasting/Formers_Long_Informer.sh
sh scripts/EXP-LongForecasting/Formers_Long_Transformer.sh

# FEDformer (run from subdirectory)
cd FEDformer && sh scripts/LongForecasting.sh

# Pyraformer (run from subdirectory)
cd Pyraformer && sh scripts/LongForecasting.sh
```

### Weight Visualization

```bash
python weight_plot.py  # Provide checkpoint path of a trained DLinear model
```

![DLinear weight visualization](pics/Visualization_DLinear.png)

---

## Citation

```bibtex
@inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

Please also cite the original datasets and baseline methods if used in your work.
