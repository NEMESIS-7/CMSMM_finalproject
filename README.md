# Can We Trust the Confidence?
## Per-Class Calibration of Twi and Hausa Hate-Speech Classifiers Under Class Imbalance

**Student:** Reynolds Elikem Fenuku  
**Student ID:** 22032896  
**Course:** DCIT 316 — Responsible Social Media Mining for Public Good  
**Repository:** [https://github.com/NEMESIS-7/CMSMM_finalproject](https://github.com/NEMESIS-7/CMSMM_finalproject)  
**Dataset:** [AfriHate](https://huggingface.co/datasets/afrihate/afrihate) (Muhammad et al., NAACL 2025)  
**Target Languages:** Twi (`twi`, Ghana) & Hausa (`hau`, Nigeria / West Africa)  
**Model:** TF-IDF (Word + Character n-grams) + Balanced Logistic Regression  
**Primary Code Artifact:** [`calibration_study.ipynb`](calibration_study.ipynb) (Deterministic, `SEED=42`)

---

## Executive Summary

Automated content moderation systems do not merely output class predictions (`Hate`, `Abuse`, `Normal`); they attach **confidence scores** that guide human triage. High-confidence predictions are typically automated, while low-confidence predictions are escalated to human moderators. If a classifier is **confidently wrong**, it introduces a dangerous failure mode termed **"moderator blinding"**: the system quietly files its worst errors under "trust me," actively deflecting human review away from the exact mistakes that need catching.

This project investigates per-class confidence calibration under severe class imbalance across two West African languages with inverted class distributions:
- **Twi** ($N=698$ test): Heavily dominated by **Abuse** ($70.5\%$), with rare **Normal** ($15.2\%$) and **Hate** ($14.3\%$).
- **Hausa** ($N=1,049$ test): Dominated by **Normal** ($56.7\%$), with **Abuse** ($33.5\%$) and rare **Hate** ($9.8\%$).

### Central Finding: The Calibration Trap
While post-hoc **temperature scaling** improves aggregate Expected Calibration Error (ECE) across both languages, on Twi it **worsens calibration for the rare Normal (innocuous speech) class by +42%** (ECE increases from $0.064$ to $0.091$), while substantially improving the dominant Abuse class. Because the validation loss is overwhelmed by majority Abuse instances, scalar temperature scaling sharpens logits to satisfy the majority, causing the model to become even *more* confidently mistaken when it mislabels ordinary Twi speech as toxic.

Furthermore, **39.6% of all Twi Normal-class errors** occur at high confidence ($\ge 0.70$), proving that automated triage would systematically over-censor innocent speech in Ghana without human review.

---

## Key Results at a Glance

### 1. Classification Performance (Test Split)

| Language | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Twi** | Abuse | 0.87 | 0.98 | **0.92** | 492 |
| | Hate | 0.56 | 0.70 | **0.62** | 100 |
| | Normal | 0.75 | 0.14 | **0.24** | 106 |
| | **Macro Avg** | **0.73** | **0.61** | **0.59** | **698** |
| **Hausa** | Abuse | 0.75 | 0.72 | **0.74** | 351 |
| | Hate | 0.82 | 0.53 | **0.65** | 103 |
| | Normal | 0.80 | 0.87 | **0.83** | 595 |
| | **Macro Avg** | **0.79** | **0.71** | **0.74** | **1,049** |

*Accuracy (Twi: 81%, Hausa: 79%) is omitted as a headline metric because a naive majority predictor on Twi already scores 70.5% while detecting 0% of minority instances.*

---

### 2. Per-Class ECE Before and After Temperature Scaling

Evaluated with 10 equal-mass (adaptive) bins and 95% bootstrap confidence intervals ($B=1,000$ iterations):

| Language | Optimal $T$ | Class | ECE Before | ECE After | Delta | Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Twi** | $0.7106$ | Abuse | 0.0848 | 0.0426 | $-0.042$ | Improved (Dominant class) |
| | | Hate | 0.0828 | 0.0604 | $-0.022$ | Improved |
| | | **Normal** | **0.0643** | **0.0910** | **+0.027** | **Worsened by +42% (Over-censorship risk)** |
| | | **Macro Avg** | **0.0773** | **0.0647** | **$-0.013$** | **Aggregate falsely masks minority regression** |
| **Hausa** | $0.7907$ | Abuse | 0.0415 | 0.0361 | $-0.005$ | Improved |
| | | Hate | 0.0243 | 0.0109 | $-0.013$ | Improved |
| | | Normal | 0.0468 | 0.0219 | $-0.025$ | Improved |
| | | **Macro Avg** | **0.0375** | **0.0230** | **$-0.015$** | **Uniform improvement across all classes** |

---

### 3. Moderator Blinding Quantity ($\tau = 0.70$)

Fraction of class errors made with maximum confidence $\ge 0.70$:

| Language | Class | Total Errors | Confident Errors | Blinding Rate | Operational Consequence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Twi** | Abuse | 11 | 5 | **45.5%** | Highly confident on rare Abuse misses |
| | Hate | 30 | 10 | **33.3%** | 1 in 3 hate speech misses auto-cleared |
| | **Normal** | 91 | 36 | **39.6%** | **36 innocent posts auto-censored silently** |
| **Hausa** | Abuse | 98 | 26 | **26.5%** | Moderately hedged error distribution |
| | Hate | 48 | 15 | **31.2%** | Moderate triage risk |
| | Normal | 79 | 26 | **32.9%** | Symmetric Abuse/Normal boundary confusion |

---

## Repository Structure

```text
├── FINAL_PROJECT_22032896_Reynolds_Elikem_Fenuku.pdf  # Final research manuscript (PDF)
├── FINAL_PROJECT_22032896_Reynolds_Elikem_Fenuku.docx # Final research manuscript (Word)
├── FINAL_PROJECT_SLIDES_22032896_Reynolds_Elikem_Fenuku.pdf # Presentation slides (PDF)
├── FINAL_PROJECT_SLIDES_22032896_Reynolds_Elikem_Fenuku.pptx# Presentation slides (PowerPoint)
├── calibration_study.ipynb                            # Complete, reproducible Jupyter Notebook (SEED=42)
├── class_distribution.png                             # Visualization of Twi vs. Hausa class balance
├── confusion_classical.png                            # Normalized confusion matrices for both languages
├── reliability_twi_classical.png                      # Twi reliability curves (before/after calibration)
├── reliability_hau_classical.png                      # Hausa reliability curves (before/after calibration)
├── triage_risk_coverage_twi.png                       # Twi triage risk-coverage trade-off curves
├── triage_risk_coverage_hausa.png                     # Hausa triage risk-coverage trade-off curves
└── requirements.txt                                   # Python dependencies
```

---

## Experimental Methodology

1. **Step 0 Pre-Verification**:
   - Confirmed true label strings via `df['label'].unique()`: `['Abuse', 'Hate', 'Normal']`.
   - Verified that all test cells exceeded the minimum statistical viability threshold ($N \ge 30$).
2. **Text Preprocessing**:
   - Minimal cleaning: lowercasing, stripping URLs (`https?://\S+`) and `@handles`, and whitespace normalization. Avoided aggressive stemming or European-centric tokenizers that distort African language morphology.
3. **Feature Pipeline & Model Scope**:
   - Concatenated sublinear TF-IDF word n-grams ($1\text{--}2$) and sublinear character n-grams ($2\text{--}4$ within word boundaries) to capture agglutinative and morphologically rich syntax.
   - **Scope Note on Transformer**: While the original proposal envisioned contrasting the classical baseline against a fine-tuned multilingual transformer (`Davlan/afro-xlmr-base`), the transformer was formally omitted as a documented scope reduction due to local compute constraints (CPU-only PyTorch environment and excessive training latency). The core research contributions—per-class ECE, temperature scaling inversion, and moderator blinding—are fully preserved using the classical model (see `REPORT.md` §6.4).
4. **Calibration Measurement**:
   - Hand-rolled one-vs-rest Expected Calibration Error (ECE) using 10 equal-mass (adaptive) quantiles to prevent empty-bin artifacts common in imbalanced data.
   - Non-parametric 95% bootstrap confidence intervals ($B=1,000$).
5. **Mitigation**:
   - Post-hoc Temperature Scaling via L-BFGS-B / bounded scalar minimization on validation set cross-entropy loss.

---

## Getting Started & Reproducibility

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd final_project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter lab calibration_study.ipynb
```
The notebook executes deterministically with `SEED=42`. It automatically fetches the `afrihate` subsets from Hugging Face via the `datasets` library and reproduces all metrics and figures.

---

## Literature Matrix

| Reference | Domain / Core Focus | Key Findings & Theoretical Insights | Direct Relevance to this Study |
| :--- | :--- | :--- | :--- |
| **Guo et al. (ICML 2017)** | Calibration of modern neural networks | Established Expected Calibration Error (ECE) and reliability diagrams; introduced post-hoc scalar Temperature Scaling. | Supplies our formal definition of ECE and temperature scaling mitigation framework. |
| **Ahuja et al. (EMNLP 2022)** | Calibration of massively multilingual models | Multilingual models degrade in calibration for low-resource and typologically distant languages despite acceptable accuracy. | Directly motivates empirical calibration measurement on African languages (Twi/Hausa) rather than assuming cross-lingual transfer. |
| **Islam et al. (2021)** | Calibration under long-tailed class imbalance | Uniform scalar temperature scaling fails under severe class imbalance; majority class dominates validation loss, hurting minority calibration. | Provides theoretical grounding for per-class ECE measurement and explains the Twi Normal scaling failure. |
| **Muhammad et al. (NAACL 2025)** | AfriHate African hate speech benchmark | First native-annotated hate-speech collection across 15 African languages; documents severe class imbalance and bidirectional moderation risks. | Supplies our dataset, ground-truth label definitions, and baseline framing. |

---

## Data Card & Model Card Summary

Detailed Data and Model Cards are permanently documented in the technical manuscript ([`REPORT.md`](REPORT.md) Appendices A & B):

- **Data Card Summary ([`REPORT.md` Appendix A](REPORT.md#appendix-a--data-card))**:
  - Source: Hugging Face `afrihate/afrihate` (Subsets: `twi`, `hau`).
  - Unit of Analysis: Post (tweet).
  - Licence: Tagged Apache 2.0; handled conservatively (derived metrics and aggregate tables only; no raw tweet text redistributed).
  - Test Splits: Twi ($N=698$: Abuse=492, Hate=100, Normal=106); Hausa ($N=1,049$: Abuse=351, Hate=103, Normal=595).
  - Sampling Caveat: Compiled via keyword-and-handle queries; class proportions reflect sampling filters, not societal base rates.
- **Model Card Summary ([`REPORT.md` Appendix B](REPORT.md#appendix-b--model-card))**:
  - Architecture: TF-IDF (Word 1–2 grams + Char 2–4 grams within word boundaries) + Balanced Logistic Regression (`class_weight='balanced'`).
  - Intended Use: **Diagnostic research instrument only.**
  - **Operational Prohibition**: **NOT suitable for live deployment** in any automated content moderation pipeline. Live deployment would systematically censor innocent Ghanaian speech due to the high Normal blinding rate (39.6%).

---

## Failure Log & Scientific Boundaries

Documenting failure modes and negative results is essential for responsible AI research:

1. **Failure Mode 1: The Calibration Trap of Temperature Scaling**
   - *Outcome:* Applying post-hoc temperature scaling on Twi reduced aggregate ECE ($0.0773 \to 0.0647$), but worsened minority **Normal ECE by +42%** ($0.0643 \to 0.0910$).
   - *Root Cause:* Validation NLL is 70% dominated by majority Abuse instances. The scalar optimizer fits $T=0.7106 < 1$, which sharpens output probabilities globally, making the model substantially *more overconfident* when it mislabels innocent Normal posts as toxic.
2. **Failure Mode 2: Twi Normal Class Recall Collapse**
   - *Outcome:* The classical model achieved an F1-score of only 0.24 on Twi Normal, detecting only 14% of innocent speech (Recall = 0.14) while misclassifying 86% of Normal posts into Abuse or Hate.
   - *Mitigation Attempted:* Balanced class weights prevented gradient collapse, but lexical surface cues in Twi overlap heavily between Normal and abusive social speech under TF-IDF features.
3. **Failure Mode 3: Reversal of the Rare-Class Miscalibration Hypothesis**
   - *Initial Hypothesis:* We hypothesized that miscalibration would track the rare class (Twi Normal and Hausa Hate would suffer highest ECE).
   - *Empirical Finding:* In both languages, the **dominant class carried the highest ECE** (Abuse in Twi: 0.0848; Normal in Hausa: 0.0468) because abundant training data inflated output confidence beyond empirical accuracy.
4. **Scope Decision: Transformer Computation Constraint**
   - *Planned Scope:* Contrast classical model against fine-tuned multilingual transformer (`Davlan/afro-xlmr-base`).
   - *Scope Reduction:* Omitted due to local CPU-only PyTorch hardware constraints and multi-hour training latencies. The core research questions (per-class ECE, temperature scaling inversion, and moderator blinding) are fully established using the deterministic classical baseline.

---

## Reproducibility Checklist

- **Deterministic Random Seed:** Fixed `SEED = 42` across NumPy, scikit-learn, and dataset splitting.
- **Executable Code Artifact:** Primary notebook [`calibration_study.ipynb`](calibration_study.ipynb) contains visible outputs and execution counts.
- **Clean Dependency Manifest:** Pinned dependencies listed in [`requirements.txt`](requirements.txt).
- **Public Benchmark Access:** Data fetched directly from Hugging Face `afrihate/afrihate` using official splits (`train`, `validation`, `test`).
- **Step 0 Verification Passed:** Label strings confirmed (`['Abuse', 'Hate', 'Normal']`); test cell support verified ($N \ge 30$).
- **No Raw Data Leakage:** Zero tweet text committed to version control; only derived metrics and plots published.
- **Deliverables Aligned:** Research manuscript and oral defence presentation slides available in both PDF and editable formats.

---

## Ethics, Privacy, and Responsible AI Disclosures

- **No Raw Data Redistribution**: Under strict compliance with dataset ethics and conservative licensing principles, **no raw tweet text** is committed to this repository or printed in notebook outputs. Only derived aggregate statistics, confusion matrices, and calibration plots are retained.
- **Diagnostic Instrument Only**: The models in this repository are **weak baseline diagnostic instruments** constructed to study calibration dynamics under severe skew. They are **NOT suitable for live deployment** in content moderation systems.
- **Base Rate Caveat**: AfriHate was compiled via keyword-and-handle sampling. The class proportions reflect sampling choices, not the real-world base rate of hate speech in Ghanaian or Nigerian online spaces.
- **AI Disclosure**: AI programming assistants (Claude / Anthropic, Gemini / Google) were used for code scaffolding, plotting utilities, and technical documentation drafting. All empirical results, figures, and calculations were independently verified against deterministic notebook execution runs.
