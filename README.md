# AD-FTD: Adversarial Disclosure-Aware Financial Statement Fraud Detection

**Paper:** "Adversarial Disclosure-Aware Financial Statement Fraud Detection via Counterfactual Trajectory Learning"  
*Submitted to IEEE Transactions on Big Data*

## Overview

AD-FTD addresses a fundamental weakness of existing trajectory-based fraud detectors: they assume fraud manifests as *anomalous* disclosure patterns.  Modern fraudsters increasingly engineer disclosures to appear *statistically normal*, making anomaly-based detectors ineffective.

AD-FTD reframes detection as identifying **unnatural consistency** by jointly modelling:
1. The **observed** disclosure trajectory Z_t
2. Its **counterfactual expectation** Ẑ_t (what an honest firm *would* have written)
3. The **deviation** ΔZ_t = Z_t − Ẑ_t

The three-part joint representation `[Z ‖ Ẑ ‖ ΔZ]` is fed to a TCN+BiLSTM classifier trained with FGSM adversarial regularisation.

## Architecture

```
MD&A text (t-1, t)
       │
  ParaEmb (GPT-3.5 / Llama-3-8B)
       │
  FraudW2V (9 word categories × 4 change types → Z_t ∈ R^36)
       │
       ├──────────────────────────────────────┐
       │                                      │
  CounterfactualGenerator (g_φ)         FraudDetector (f_θ)
  TCN + feature fusion                  TCN + BiLSTM + sigmoid
  (hist. traj + fin. ratios + peers)         │
       │                                      │
       └──────── [Z ‖ Ẑ ‖ ΔZ] ──────────────►│
                                         P(fraud)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
# For Llama-3 local embedding (no API cost):
pip install -e ".[llama]"
```

### 2. Prepare data

| File | Source |
|------|--------|
| `data/raw/aaer_labels.csv` | [SEC AAER records](https://www.sec.gov/divisions/enforce/friactions.htm) |
| `data/raw/compustat_ratios.csv` | Compustat via WRDS |
| EDGAR text | Downloaded automatically from [HuggingFace](https://huggingface.co/datasets/eloukas/edgar-corpus) |

Required CSV columns:
- `aaer_labels.csv`: `cik, fyear, misstatement_start, misstatement_end, aaer_num`
- `compustat_ratios.csv`: `cik, gvkey, fyear, sic, at, [ratio columns...]`

### 3. Preprocess (one-time, ~72h with GPT-3.5 / ~60h with Llama-3)

```bash
# GPT-3.5 (~USD 3,500 batch cost)
export OPENAI_API_KEY=sk-...
python scripts/preprocess.py --backend openai

# Llama-3-8B-Instruct 4-bit (free, requires 24GB VRAM)
export HF_TOKEN=hf_...
python scripts/preprocess.py --backend llama3
```

### 4. Train

```bash
python scripts/train.py \
    --features data/features/samples.pkl \
    --epochs 50 \
    --device cuda
```

Runs 4-fold CV × 10 resampling iterations (40 evaluation runs total).  
Checkpoints saved to `checkpoints/adftd_r{resample}_f{fold}.pt`.

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/adftd_r0_f0.pt \
    --features data/features/samples.pkl
```

Reports: Precision, Recall, Macro F1, PRC-AUC, ROC-AUC, ARS, Expected Cost at c_FN/c_FP ∈ {100, 500, 1000}.

## Project Structure

```
AD-FTD/
├── src/adftd/
│   ├── config.py              # All hyperparameters
│   ├── data/
│   │   ├── edgar_loader.py    # EDGAR-CORPUS loader
│   │   ├── aaer_matcher.py    # AAER fraud label matcher
│   │   ├── peer_group.py      # 2-digit SIC + asset decile peer groups
│   │   └── dataset.py         # PyTorch Dataset + sample builder
│   ├── features/
│   │   ├── embedders.py       # GPT-3.5 and Llama-3 paragraph embedders
│   │   ├── fraud_w2v.py       # 9-category FraudW2V scoring
│   │   └── trajectory.py      # Paragraph alignment + change trajectory
│   ├── models/
│   │   ├── tcn.py             # Dilated causal TCN
│   │   ├── counterfactual.py  # Counterfactual generator g_φ
│   │   ├── detector.py        # TCN+BiLSTM fraud detector f_θ
│   │   └── adftd.py           # Full AD-FTD model
│   ├── training/
│   │   ├── losses.py          # L = L_cls + α·L_dev + β·L_adv
│   │   ├── fgsm.py            # Embedding-space FGSM perturbation
│   │   └── trainer.py         # 4-fold CV trainer with early stopping
│   └── evaluation/
│       └── metrics.py         # PRC-AUC, F1, ARS, Expected Cost
├── scripts/
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── requirements.txt
├── requirements-llama.txt     # Extra deps for local Llama-3 embedder
└── pyproject.toml
```

## Key Design Choices

### Peer-group construction
Industry = 2-digit SIC code; size = decile rank of total assets within SIC × fiscal year cell.  
Minimum 5 firms required; falls back to 1-digit SIC super-group.  
Peer groups are built **only from training data** to prevent look-ahead bias.

### Evaluation metrics
- **Primary:** PRC-AUC and Macro F1 (robust under 1:97 class imbalance)
- **Secondary:** ROC-AUC (reported for prior-work comparability only)
- **Cost-sensitive:** Expected Cost at multiple c_FN/c_FP ratios (Elkan 2001)

### FGSM scope
FGSM operates in **embedding space** (not natural language space).  
It serves as a proxy for embedding-stability under minor representational shifts.  
Language-space strategic adversarial training using LLM-generated MD&A rewrites is ongoing work.

## Citation

```bibtex
@article{adftd2026,
  title   = {Adversarial Disclosure-Aware Financial Statement Fraud Detection
             via Counterfactual Trajectory Learning},
  journal = {IEEE Transactions on Big Data},
  year    = {2026},
  note    = {Under review}
}
```

## Data Availability

- EDGAR-CORPUS: https://huggingface.co/datasets/eloukas/edgar-corpus  
- SEC AAER records: https://www.sec.gov/divisions/enforce/friactions.htm  
- Preprocessed features, AAER-GVKEY matching, model weights: this repository  
- GPT-3.5 embeddings cannot be redistributed (OpenAI ToS); use the Llama-3 alternative.

## License

MIT
