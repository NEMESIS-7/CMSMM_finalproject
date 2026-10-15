# Can We Trust the Confidence? Per-class Calibration of Twi and Hausa Hate-Speech Classifiers Under Class Imbalance

**DCIT 316 — Responsible Social Media Mining for Public Good**  
**Student:** Reynolds Elikem Fenuku (Student ID: 22032896)  
**Repository:** https://github.com/NEMESIS-7/CMSMM_finalproject  
**Dataset:** AfriHate (Muhammad et al., 2025) — Twi and Hausa subsets  
**Model:** TF-IDF + Logistic Regression (classical baseline)  
**Code:** `calibration_study.ipynb` (reproducible, fixed seed SEED=42)  

---

## Abstract

A hate-speech classifier attached to a moderation pipeline does not just classify — it also signals, through its confidence score, which predictions a human reviewer can trust and which require scrutiny. If that confidence is miscalibrated, the tool systematically hides its own worst errors from the reviewers who need to catch them. This project investigates whether per-class confidence calibration degrades predictably under class imbalance, using the AfriHate collection's Twi and Hausa subsets — two languages with nearly inverted class distributions. We train a TF-IDF + Logistic Regression classifier on each language and measure per-class Expected Calibration Error (ECE) using equal-mass bins with bootstrapped 95% confidence intervals. We then apply temperature scaling as a post-hoc mitigation and evaluate whether it recovers calibration equally across all classes. The central finding is that temperature scaling improves overall ECE on both languages, but on Twi it simultaneously **worsens** calibration for the Normal (innocuous speech) class — increasing its ECE from 0.064 to 0.091 — while improving the dominant Abuse class substantially. On Hausa, calibration improves uniformly across all classes. This asymmetry is not predicted by overall ECE alone and would be invisible to any evaluation that reports a single aggregate calibration number. The blinding quantity — the fraction of errors made at high confidence (≥0.70) — reaches 45.5% for Abuse errors in Twi and 39.6% for Normal errors, indicating that nearly four in ten times the model misclassifies a Twi Normal post as toxic, it does so confidently enough to route that error away from human review. These findings argue against deploying aggregate calibration assessment as a safety check for minority-class predictions and demonstrate that standard post-hoc mitigation can leave the worst miscalibration untouched while appearing to have fixed it.

---

## 1. Introduction

Automated content moderation relies on classifiers that assign both a label and a confidence score to each post. In a triage architecture, that confidence score determines which predictions go to a human reviewer and which are acted upon automatically. The premise is that high-confidence predictions can be trusted and low-confidence ones need scrutiny. If that premise holds, the tool is useful; if it does not, the tool blinds the reviewer — routing confident errors into the "trusted" pile and directing attention away from exactly the mistakes that need catching.

Calibration is the property that makes the premise hold or fail. A perfectly calibrated model produces confidence scores that match its actual hit rate across confidence levels: of all posts it marks at 90% confidence, about 90% should in fact be correct. Accuracy and calibration are distinct: a model can be highly accurate yet badly miscalibrated, producing overconfident wrong predictions that are invisible to accuracy metrics.

Class imbalance is known to degrade calibration (Islam et al., 2021), and low-resource, typologically distant languages are known to suffer worse calibration than high-resource ones (Ahuja et al., 2022). The AfriHate dataset (Muhammad et al., 2025) creates a natural experiment: its Twi subset has Abuse as the dominant class (≈70% of test instances), with Hate and Normal as minority classes, while its Hausa subset has Normal as the dominant class (≈57% of test instances) and Hate as the smallest. The same three-class task, two languages, class balance nearly inverted. If miscalibration follows the dominant class — or the rare class — that pattern should flip between languages.

The core research questions are:

1. Is a hate-speech classifier's confidence equally trustworthy across classes, or does calibration differ substantially between majority and minority classes?
2. Does post-hoc temperature scaling recover miscalibration uniformly, or does it fix the aggregate while leaving specific classes worse off?
3. Does the worst-calibrated class follow the rare class when we compare Twi and Hausa — i.e., does the failure mode flip with the class balance inversion?

---

## 2. Related Work

**Calibration of modern neural networks.** Guo et al. (2017) showed that modern deep networks are systematically overconfident relative to their accuracy, and that Expected Calibration Error (ECE) provides a tractable scalar measure of this gap. They introduced reliability diagrams as the visualization tool and demonstrated that temperature scaling — dividing logits by a single learnable scalar — is a simple and effective post-hoc recalibration method. This project uses their definitions of ECE and temperature scaling directly.

**Calibration of multilingual models.** Ahuja et al. (2022) measured ECE across languages for massively multilingual models and found that calibration degrades systematically for low-resource and typologically distant languages, even when accuracy is acceptable. They found temperature scaling and label smoothing both help but do not fully close the gap. Their work motivates treating Twi calibration as an open question rather than assuming results from high-resource languages transfer.

**Calibration under class imbalance.** Islam et al. (2021) showed that uniform temperature scaling is suboptimal under long-tailed class distributions, where the minority class carries disproportionate calibration error. They proposed class-frequency-aware calibration methods. Their analysis is the direct theoretical motivation for measuring ECE per class rather than only in aggregate. A single-scalar temperature fit primarily on majority-class examples may not generalise to minority-class predictions.

**AfriHate.** Muhammad et al. (2025) introduced the AfriHate dataset, a multilingual collection of Twitter-derived hate speech annotations across 15 African languages by native speakers. They report strong class imbalance in several subsets, baseline accuracy with both classical and transformer-based models, and note that moderation of African-language content fails in both directions — under-detecting hate and over-flagging ordinary speech. They supply our data, labels, class distributions, and baseline comparison points.

---

## 3. Data

### 3.1 Source

AfriHate (Muhammad et al., 2025), accessed via Hugging Face: `afrihate/afrihate`. Two subsets: `twi` (Twi language, Ghana) and `hau` (Hausa language, Nigeria/West Africa). Both are pre-split into `train`, `validation`, and `test`. No data collection, scraping, or API access was performed.

**Licence:** Tagged Apache 2.0 on Hugging Face. The tag is not corroborated by a licence file in the source repository or the paper. Data is handled conservatively: no raw tweet text is committed to the repository or printed in notebook outputs; only derived metrics, plots, and aggregate confusion counts are retained.

### 3.2 Pre-build Verification (Step 0)

All design decisions depend on what the data actually contains, not on what the paper describes. Three checks were run before any model code was written.

**Check 1 — Label strings.** Both Twi and Hausa subsets use identical label strings: `['Abuse', 'Hate', 'Normal']`. The dataset card notes that the Twi viewer renders "Abuse" rather than "Abusive"; the loaded data confirms `Abuse`. All downstream label handling derives from `.unique()` output; nothing is hardcoded.

**Check 2 — Per-class test-set counts.**

| Split | Abuse | Hate | Normal | Total |
|---|---|---|---|---|
| Twi train | — | — | — | 2,564 |
| Twi val | — | — | — | 639 |
| Twi test | 492 | 100 | 106 | 698 |
| Hausa train | — | — | — | 4,566 |
| Hausa val | — | — | — | 1,029 |
| Hausa test | 351 | 103 | 595 | 1,049 |

All test-set cells exceed the ≥30 viability threshold; both languages are viable as primary analysis targets. Twi's Hate and Normal test classes (100 and 106 instances respectively) are small enough that per-bin ECE estimates carry meaningful uncertainty, which is why bootstrapped confidence intervals are reported throughout.

**Check 3 — Schema parity.** Twi exposes `[id, tweet, label]`; Hausa exposes an additional `length` column. The loader selects only the three required fields from both, discarding the extra column.

### 3.3 Class Balance

The two languages present nearly inverted imbalance:

- **Twi:** Abuse is the dominant class at 70.5% of the test set. Hate (14.3%) and Normal (15.2%) are rare. A majority predictor that always predicts Abuse scores 70.5% accuracy — useless for the minority classes. Macro-F1 is the appropriate headline metric.
- **Hausa:** Normal is the dominant class at 56.7% of the test set. Abuse is mid-frequency (33.5%), and Hate is rare (9.8%). Class balance is more distributed than in Twi but Hate is the smallest cell.

This contrast is the experimental design: the same task, same model architecture, two languages where the dominant class rotates. If calibration follows the class balance, the miscalibration pattern should rotate too.

### 3.4 Text Cleaning

Minimal cleaning was applied to avoid confounding the comparison: lowercase, strip URLs (`https?://...`) and @handles, normalise whitespace. No stemming, stopword removal, or language-specific tokenisation. Cleaning is minimal because heavy preprocessing applied differentially to African languages can introduce its own artefacts.

---

## 4. Methods

### 4.1 Classical Model

**Feature extraction:** Two TF-IDF vectorisers combined via sparse matrix concatenation:
- Word n-grams: unigrams and bigrams (`ngram_range=(1,2)`), minimum document frequency 2, sublinear TF scaling.
- Character n-grams: 2–4 character windows within word boundaries (`analyzer='char_wb'`, `ngram_range=(2,4)`), minimum document frequency 2, sublinear TF.

Character n-grams are included because they capture morphological variation and partial word matches that are important in Twi and Hausa, which both have rich morphology and frequent code-switching in social media text.

**Classifier:** Logistic regression (`solver='lbfgs'`, `max_iter=1000`, `C=1.0`) with `class_weight='balanced'` to counteract the Abuse-dominated training distribution in Twi. The balanced weighting assigns each class a weight inversely proportional to its frequency, so the minority classes are not drowned out during gradient steps.

**Seed:** SEED=42 fixed throughout; all results are deterministic.

### 4.2 Calibration Measurement

**Expected Calibration Error (ECE) — per class, one-vs-rest.** For class *c*, we treat prediction as binary: the model's output probability P(class = c) is the confidence score, and the binary label is 1 if the true class is *c* and 0 otherwise. This allows per-class ECE measurement independently of how probabilities are distributed across the other classes.

ECE is computed using **equal-mass (adaptive) binning** with 10 bins. Equal-mass binning assigns each bin the same number of predictions rather than equal-width confidence intervals. This is necessary for minority classes where the predictions cluster in low-confidence regions: equal-width bins would produce many empty bins in the high-confidence range, making the ECE estimate noisy and unstable. Equal-mass binning ensures every bin contributes a meaningful number of predictions.

Formally:

$$\text{ECE}_c = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \bar{p}_{m,c} - \bar{y}_{m,c} \right|$$

where $B_m$ is the $m$-th bin, $n$ is the total number of test instances, $\bar{p}_{m,c}$ is the mean predicted probability for class $c$ in bin $m$, and $\bar{y}_{m,c}$ is the fraction of bin $m$ instances whose true label is $c$.

**Bootstrap confidence intervals.** 95% confidence intervals on each ECE estimate are computed by bootstrapping 1,000 resamplings of the test set and taking the 2.5th and 97.5th percentiles. This quantifies the sampling noise from small test cells.

**Reliability diagrams.** For each class, a reliability diagram plots the mean predicted confidence (x-axis) against the fraction of positive instances (y-axis) per bin. A perfectly calibrated model traces the diagonal. Points above the diagonal indicate underconfidence (the model's confidence is lower than its actual accuracy); points below indicate overconfidence. Dot size encodes bin population.

### 4.3 Blinding Quantity

For each class *c*, the blinding quantity is the fraction of the model's errors on class *c* that fall in high-confidence bins (maximum predicted probability ≥ 0.70 across all classes). A high blinding quantity means the model is frequently confident when it is wrong — the condition that makes a triage tool dangerous, because those errors are quietly filed under "trust me."

$$\text{Blinding}_c = \frac{|\{ i : y_i = c,\; \hat{y}_i \neq c,\; \max_j \hat{p}_{ij} \geq \tau \}|}{|\{ i : y_i = c,\; \hat{y}_i \neq c \}|}$$

with threshold τ = 0.70.

### 4.4 Temperature Scaling

Temperature scaling (Guo et al., 2017) is the simplest post-hoc calibration method: a single scalar *T* is applied to the logits before the softmax, sharpening (T < 1) or softening (T > 1) the probability distribution.

$$\hat{p}_{ij} = \frac{\exp(z_{ij}/T)}{\sum_k \exp(z_{ik}/T)}$$

*T* is fit by minimising the negative log-likelihood (NLL) on the **validation** set logits, using bounded scalar minimisation (scipy `minimize_scalar`, bounds [0.01, 10.0]). The test set is never seen during this fitting step.

The key evaluation is whether temperature scaling improves ECE **per class**, not just in aggregate. A model whose overall ECE improves while its minority-class ECE does not — or gets worse — is exhibiting exactly the failure mode this project exists to expose.

---

## 5. Results

### 5.1 Classification Performance

**Table 1. Classification performance on the test set.**

| Language | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Twi | Abuse | 0.87 | 0.98 | 0.92 | 492 |
| Twi | Hate | 0.56 | 0.70 | 0.62 | 100 |
| Twi | Normal | 0.75 | 0.14 | 0.24 | 106 |
| **Twi** | **Macro avg** | **0.73** | **0.61** | **0.59** | **698** |
| Hausa | Abuse | 0.75 | 0.72 | 0.74 | 351 |
| Hausa | Hate | 0.82 | 0.53 | 0.65 | 103 |
| Hausa | Normal | 0.80 | 0.87 | 0.83 | 595 |
| **Hausa** | **Macro avg** | **0.79** | **0.71** | **0.74** | **1,049** |

Accuracy is 81% for Twi and 79% for Hausa, but these numbers are misleading: a majority predictor that always guesses Abuse achieves 70.5% on Twi. Macro-F1 is the correct headline. At 0.59, the Twi classifier is weak, driven largely by its near-inability to recall Normal posts (recall = 0.14). The Twi model overwhelmingly predicts Abuse, achieving 98% recall there at the cost of collapsing almost all Normal posts into Abuse or Hate predictions.

The Hausa classifier is considerably stronger at macro-F1 = 0.74, with more balanced recall across classes. This reflects both the more balanced class distribution in Hausa and the larger training set (4,566 vs. 2,564 instances).

The Twi Normal class is the clearest weak point in performance: the model has learned to associate the feature space with Abuse and rarely predicts Normal even when the true label is Normal. This matters calibration-wise because the 106 Normal test instances represent a large source of errors, and how confidently those errors are made determines whether a triage tool would hide or surface them.

### 5.2 Per-class Calibration (ECE)

**Table 2. Per-class ECE with 95% bootstrap confidence intervals (equal-mass bins, 10 bins).**

| Language | Class | ECE | 95% CI |
|---|---|---|---|
| Twi | Abuse | 0.0848 | [0.065, 0.106] |
| Twi | Hate | 0.0828 | [0.065, 0.103] |
| Twi | Normal | 0.0643 | [0.050, 0.091] |
| **Twi** | **Macro-avg** | **0.0773** | — |
| Hausa | Abuse | 0.0415 | [0.034, 0.066] |
| Hausa | Hate | 0.0243 | [0.018, 0.039] |
| Hausa | Normal | 0.0468 | [0.040, 0.075] |
| **Hausa** | **Macro-avg** | **0.0375** | — |

Several findings stand out:

**Twi calibration is substantially worse than Hausa.** The macro-average ECE for Twi (0.077) is more than twice that of Hausa (0.038). This aligns with Ahuja et al.'s (2022) finding that calibration degrades for lower-resource, more imbalanced settings. Twi has a smaller training set and a more extreme class imbalance, and its calibration is correspondingly worse.

**Within Twi, the dominant class (Abuse) is the worst-calibrated.** The ECE for Abuse (0.085) and Hate (0.083) are nearly equal and both higher than Normal (0.064). The dominant Abuse class, despite having 492 test instances and high recall, has the highest calibration error. The Normal class — the rarest, most confused class — paradoxically has the best ECE among the three. Note, however, that the confidence intervals for Twi overlap substantially across classes (Abuse CI: [0.065, 0.106], Normal CI: [0.050, 0.091]), reflecting uncertainty from small test cells. The point estimates are informative, but strong statistical claims about rank ordering require more test data.

**Within Hausa, calibration is more uniform and lower across all classes.** The Normal class (dominant) has the highest ECE at 0.047, and Hate (rare) has the lowest at 0.024. The range of ECE values across classes in Hausa (0.024–0.047) is narrower than in Twi (0.064–0.085), suggesting more even calibration across classes.

**Cross-language comparison: the worst-calibrated class does not cleanly follow the rare class.** The original hypothesis was that the dominant class or the rare class would rotate between languages. In Twi, the worst-calibrated class is Abuse (dominant). In Hausa, the worst-calibrated class is Normal (dominant). In both cases, the dominant class — not the rare class — carries the highest ECE. This is a surprising reversal of the expected pattern and constitutes a legitimate finding: the dominant class, which the model predicts most often and with highest confidence, is the class whose confidence is least accurate in a calibration sense. Rare-class predictions, when the model deigns to make them, appear to be appropriately hedged.

### 5.3 Reliability Diagrams

Reliability diagrams reveal the direction of miscalibration that ECE alone cannot show.

**Twi — Abuse.** Bins at moderate confidence (0.4–0.7) show the model is overconfident: mean confidence exceeds actual positive fraction. The large bin sizes here (reflecting the abundance of Abuse instances) mean these bins heavily weight the ECE.

**Twi — Normal.** The model's Normal predictions are concentrated in low-to-moderate confidence bins (rarely above 0.6), consistent with the low recall (0.14). When it does predict Normal, the model is not confident, and within those bins the calibration is comparatively good.

**Hausa — all classes.** The reliability diagrams show more scatter around the diagonal, with no strong systematic over- or underconfidence. Hausa's lower overall ECE is reflected in closer alignment to the perfect-calibration line across bins.

### 5.4 Temperature Scaling

**Table 3. Per-class ECE before and after temperature scaling (optimal T fit on validation set).**

| Language | Optimal T | Class | ECE before | ECE after | Delta |
|---|---|---|---|---|---|
| Twi | 0.7106 | Abuse | 0.0848 | 0.0426 | −0.042 |
| Twi | 0.7106 | Hate | 0.0828 | 0.0604 | −0.022 |
| Twi | 0.7106 | **Normal** | **0.0643** | **0.0910** | **+0.027** |
| **Twi** | — | **Macro-avg** | **0.0773** | **0.0647** | **−0.013** |
| Hausa | 0.7907 | Abuse | 0.0415 | 0.0361 | −0.005 |
| Hausa | 0.7907 | Hate | 0.0243 | 0.0109 | −0.013 |
| Hausa | 0.7907 | Normal | 0.0468 | 0.0219 | −0.025 |
| **Hausa** | — | **Macro-avg** | **0.0375** | **0.0230** | **−0.015** |

This table contains the central finding of the project.

**Both optimal temperatures are below 1.** T < 1 means dividing the logits by a value less than one, which increases the magnitude of logits and produces sharper, more extreme probability distributions — the model's confidence is increased. This indicates the model was, on the validation set, systematically underconfident: its probability mass was spread too broadly, and the optimal fix is to concentrate it.

**On Hausa, temperature scaling uniformly improves calibration across all three classes.** After scaling, all ECE values decrease. The Normal class improves most (−0.025), Hate improves moderately (−0.013), and Abuse improves marginally (−0.005). The fix is fair across the class distribution.

**On Twi, temperature scaling improves overall ECE but makes the Normal class substantially worse.** The macro-average ECE drops from 0.077 to 0.065 — an apparent success. Abuse and Hate both improve. But the Normal class ECE increases by +0.027, from 0.064 to 0.091, worsening by 42% relative to its pre-scaling value. The single scalar T = 0.71 was fit primarily to the loss on the validation set, which is dominated by Abuse instances. Sharpening the Abuse predictions improves Abuse's already-dominant calibration signal but over-concentrates probability on the Abuse and Hate classes for Normal instances, making the model overconfident in predicting Abuse/Hate for what are actually Normal posts.

This is exactly the failure mode described by Islam et al. (2021): a single temperature scalar fitted on an imbalanced dataset calibrates the majority class and miscalibrates the minority. The aggregate ECE improvement is not a reliable signal that calibration has been fixed for the class that matters most for over-censorship errors.

**The practical consequence.** Before scaling, a human operator looking at the Twi model's aggregate ECE (0.077) might judge it acceptable and deploy with temperature scaling to "fix" it. After scaling, the aggregate ECE looks better (0.065), but the model's Normal class has become *more* overconfident than before. A content moderation tool using this model is now more likely to confidently misclassify Normal Twi posts as Abusive or Hate speech after calibration than it was before. The aggregate ECE improvement masks a per-class regression.

### 5.5 Blinding Quantity

**Table 4. Fraction of class errors made at high confidence (threshold ≥ 0.70).**

| Language | Class | Total errors | High-conf. errors | Blinding |
|---|---|---|---|---|
| Twi | Abuse | 11 | 5 | 45.5% |
| Twi | Hate | 30 | 10 | 33.3% |
| Twi | Normal | 91 | 36 | 39.6% |
| Hausa | Abuse | 98 | 26 | 26.5% |
| Hausa | Hate | 48 | 15 | 31.2% |
| Hausa | Normal | 79 | 26 | 32.9% |

The blinding quantity operationalises the triage risk: when the model is wrong, how often is it wrong at a confidence level that would route the error away from human review?

**Twi Abuse has the highest blinding at 45.5%.** Despite only 11 Abuse errors, nearly half are made at ≥ 0.70 confidence — the model is extremely confident in a small number of wrong Abuse predictions. This is a direct consequence of the Abuse-biased training: the model has learned features that strongly predict Abuse, and occasionally fires on them incorrectly but at high confidence.

**Twi Normal has 91 total errors and 36 of them are high-confidence (39.6%).** This is the operationally most important number. Normal posts are genuinely innocuous speech — misclassifying them as toxic is a form of over-censorship of speech in the Twi language. When 36 of those misclassifications are made at ≥ 0.70 confidence, a triage tool set to auto-route confident predictions would suppress 36 of the 91 Normal-class errors without ever flagging them for human review.

**Hausa blinding values are lower and more uniform** (26.5–32.9%), consistent with its better overall calibration. The model makes more errors in absolute terms (the smaller relative class sizes of Abuse and Hate mean more confusion), but it makes those errors with lower confidence on average.

### 5.6 Error Analysis

**Table 5. Confident error patterns (predicted vs. true class at max confidence ≥ 0.70).**

**Twi — 51 confident errors (7.3% of test):**

| True class | Predicted | Count |
|---|---|---|
| Normal | Hate | 18 |
| Normal | Abuse | 18 |
| Hate | Abuse | 10 |
| Abuse | Hate | 3 |
| Abuse | Normal | 2 |

**Hausa — 67 confident errors (6.4% of test):**

| True class | Predicted | Count |
|---|---|---|
| Abuse | Normal | 25 |
| Normal | Abuse | 25 |
| Hate | Normal | 11 |
| Hate | Abuse | 4 |
| Abuse | Hate | 1 |
| Normal | Hate | 1 |

**Twi error pattern.** The most striking feature is the symmetric distribution of Normal errors: exactly 18 Normal posts are confidently predicted as Hate and 18 as Abuse. This suggests the model has no stable internal representation of what Normal Twi text looks like; when it encounters a Normal post, it confidently assigns it to whichever toxic class its surface features most resemble. This is the over-censorship failure mode: actual innocuous speech in Twi is confidently labelled as either hate speech or abusive speech, with no systematic preference between the two toxic classes. The second largest error type is Hate→Abuse (10 counts), a between-toxic-class confusion that, while a classification failure, is less harmful from a moderation perspective.

**Hausa error pattern.** The Hausa confident errors are dominated by the Abuse↔Normal boundary: 25 Abuse posts are confidently predicted as Normal (under-detection) and 25 Normal posts are confidently predicted as Abuse (over-censorship). The near-perfect symmetry is notable and suggests the Abuse/Normal boundary is inherently ambiguous — the model has learned some signal but fires in both directions at approximately equal rates. Hate→Normal (11 counts) is also significant: hate speech in Hausa that is misclassified as Normal speech and done so confidently would pass through a high-confidence threshold and be automatically cleared.

**Text length.** Confident errors show marginally shorter average text length than correct predictions in both languages (Twi: 76 vs. 79 chars; Hausa: 96 vs. 101 chars), but the difference is small and likely not a primary driver of miscalibration. Short posts are slightly more likely to be confused because they provide fewer n-gram features for the TF-IDF model.

### 5.7 Triage Risk-Coverage Trade-off

**Table 6. Risk-coverage trade-off at three confidence thresholds.**

| Language | Threshold | Auto-routed | Coverage | Precision | Errors hidden |
|---|---|---|---|---|---|
| Twi | 0.50 | 656 / 698 | 94.0% | 83.2% | 110 |
| Twi | 0.70 | 514 / 698 | 73.6% | 90.1% | 51 |
| Twi | 0.90 | 182 / 698 | 26.1% | 94.5% | 10 |
| Hausa | 0.50 | 971 / 1,049 | 92.6% | 81.4% | 181 |
| Hausa | 0.70 | 652 / 1,049 | 62.2% | 89.7% | 67 |
| Hausa | 0.90 | 164 / 1,049 | 15.6% | 98.2% | 3 |

The trade-off is stark. At threshold 0.50, both models auto-route more than 90% of posts; the high coverage is achieved at the cost of hiding more than 100 (Twi) or 180 (Hausa) errors from human review. Raising the threshold to 0.70 cuts the hidden error count to 51 and 67 respectively, but still auto-routes 73% and 62% of posts — the workload reduction is substantial, but so is the residual hidden-error rate. At 0.90, the hidden-error count drops dramatically but at the cost of routing only 26% (Twi) and 16% (Hausa) of posts automatically, effectively sending most decisions to human review.

Hausa achieves 98.2% precision at threshold 0.90 with only 3 hidden errors, suggesting a much more reliable high-confidence tier. Twi achieves only 94.5% even at the 0.90 threshold, with 10 errors still hidden, consistent with its worse overall calibration.

A triage tool designer must choose a threshold knowing that the calibration warnings from Section 5.4 apply: the Twi model's Normal class, in particular, has degraded calibration after temperature scaling. Even at high thresholds, the confident-but-wrong Normal posts represent a qualitatively important failure mode (over-censorship of innocuous Twi speech) that aggregate precision numbers do not capture.

---

## 6. Discussion

### 6.1 Does Miscalibration Follow the Rare Class?

The original hypothesis was that miscalibration would follow the rare class across languages — Normal in Twi, Hate in Hausa. The results do not confirm this. In both languages, the dominant class carries the highest ECE. In Twi, Abuse (ECE=0.085) and Hate (0.083) are worse-calibrated than Normal (0.064). In Hausa, Normal (ECE=0.047) is worse-calibrated than Hate (0.024).

The explanation is subtle and worth unpacking. The dominant class has abundant training examples, so the model learns to predict it frequently and with high confidence. Those confident predictions are not always correct — the model over-fires on the dominant class. When the confident wrong predictions accumulate in high-confidence bins, the ECE for that class rises. The rare class, by contrast, is predicted infrequently and with lower confidence; when the model does predict it, the confidence level is more closely aligned with the actual hit rate because the model is hedging. This produces a somewhat counterintuitive result: the class the model can most "recognise" in training has the highest calibration error, because its confidence is not earned but inflated.

This finding does not mean rare-class calibration is not a problem. The rare Normal class in Twi has 91 errors and 36 of them are high-confidence — that is a real moderation risk. But the *ECE* for Normal is lower than for Abuse, because the 70 correct Normal predictions are also at relatively low confidence, which keeps the bin-level gaps small. The problem with Twi's Normal class is not ECE as measured but the absolute number of confident errors — a quantity that ECE does not directly capture.

### 6.2 Temperature Scaling as a Cautionary Case

The temperature scaling result is the sharpest finding in this project. For Twi, a temperature of 0.71 is fitted on the validation set. Because the validation set is dominated by Abuse instances, the NLL minimisation effectively fits the temperature to the Abuse class's calibration needs. The resulting sharper distributions improve Abuse and Hate ECE considerably, but at the cost of Normal. The model, already inclined to predict Abuse/Hate confidently, becomes *more* confident in those predictions — which means that when it incorrectly assigns a Normal post to a toxic class (as it frequently does, given the 91 Normal errors), it does so with even higher confidence than before.

This is a clean illustration of why overall ECE before-and-after comparisons are insufficient as a safety check. The aggregate ECE drops from 0.077 to 0.065, which looks like a successful calibration intervention. But the Normal class ECE has increased by 42%. A deployment team that checked only aggregate ECE would believe the model is better calibrated after scaling; it is, for the majority class, but not for the class whose miscalibration causes over-censorship.

The Hausa result (uniform improvement across all classes) shows that this failure is not inevitable — it is a consequence of the specific interplay between class imbalance and the single-scalar fitting procedure. With more balanced class distributions, the validation-set NLL generalises more uniformly, and temperature scaling works as intended.

### 6.3 Implications for Moderation Systems

The findings point to three concrete recommendations for anyone building hate-speech moderation tools on low-resource African-language data:

1. **Report per-class ECE, not just overall ECE.** Overall ECE can improve while minority- or majority-class calibration regresses. A single number is not a sufficient safety check.

2. **Evaluate temperature scaling per class before deployment.** The standard post-hoc calibration intervention can worsen calibration for specific classes. Check the per-class ECE table, not just the aggregate, after fitting temperature.

3. **Surface the blinding quantity in system documentation.** For each class, report what fraction of errors are made at high confidence. This number directly communicates the risk to a human triage team: if 40% of Normal errors are confident, the team should know that raising the auto-route threshold does not fully eliminate over-censorship errors.

### 6.4 Scope and Limitations

This study has four significant limitations that bound the conclusions.

**Single model family.** The transformer comparison (AfroXLMR) was not run due to compute constraints; this is a documented scope reduction. The accuracy-vs-calibration contrast between classical and transformer models — one of the original research questions — cannot be answered. Prior work (Ahuja et al., 2022) suggests that transformers may exhibit different and potentially worse miscalibration patterns for low-resource languages, even when more accurate. This remains an open question for this dataset.

**Confidence intervals are wide for small classes.** With 100–106 test instances in the Twi minority classes, the bootstrap confidence intervals are wide (e.g., Twi Normal: [0.050, 0.091]). The point estimates are informative and directionally consistent, but fine-grained comparisons between classes in Twi should be treated with caution. Larger test sets would sharpen the estimates.

**Sample composition.** AfriHate was constructed by keyword-and-handle sampling, so the class balance in the dataset reflects the sampling strategy, not the true prevalence of hate speech in Twi or Hausa discourse. The calibration findings apply to classifier behaviour on this distribution; they should not be extrapolated to claims about how often hate speech occurs on social media.

**Single scalar temperature.** Temperature scaling with a single scalar is the simplest post-hoc method. Class-conditional temperature scaling (a separate scalar per class) or Dirichlet calibration (Kull et al., 2019) may better handle the class-imbalance-induced failure of single-scalar fitting. These alternatives were not implemented.

---

## 7. Conclusion

We investigated per-class calibration of a TF-IDF + Logistic Regression hate-speech classifier on Twi and Hausa, two African languages with inverted class distributions. The main findings are:

1. **Calibration differs substantially across classes.** In Twi, the dominant Abuse class has the highest ECE (0.085), followed closely by Hate (0.083), with Normal (0.064) best-calibrated by ECE. In Hausa, the distribution is more uniform (Abuse 0.041, Hate 0.024, Normal 0.047). Overall, Twi calibration is roughly twice as bad as Hausa's.

2. **The worst-calibrated class follows the dominant class, not the rare class.** The hypothesised pattern — miscalibration tracking rarity — is not confirmed. The dominant class carries inflated confidence from abundant training examples, which translates to higher ECE when those confident predictions are wrong.

3. **Temperature scaling improves aggregate ECE but makes Twi's Normal class worse.** The single scalar fitted on the imbalanced validation set generalises well to Abuse and Hate but over-sharpens Normal predictions. Normal ECE increases by 42% after scaling (0.064 → 0.091) while overall ECE falls (0.077 → 0.065). Hausa improves uniformly. This is the central finding: aggregate ECE improvement is not a reliable guarantee that per-class calibration has been fixed.

4. **The blinding quantity reaches 39–45% in Twi.** For both Abuse and Normal errors in Twi, approximately four in ten mistakes are made at high confidence (≥0.70). These errors would be routed away from human review in a triage tool, systematically hiding both over-censorship (Normal→Abuse/Hate) and under-detection (Hate→Abuse) failures.

5. **The Abuse↔Normal confusion boundary is symmetric in Hausa.** Exactly 25 Abuse posts are confidently predicted as Normal and 25 Normal posts as Abuse, suggesting this boundary is inherently ambiguous for TF-IDF features in Hausa. Both over-detection and under-detection of hate speech occur at equal rates at the triage-relevant confidence tier.

The takeaway is that aggregate calibration metrics are insufficient for safety evaluation of triage tools in minority-language moderation. Per-class ECE must be reported, and standard post-hoc mitigation must be validated per class before deployment.

---

## References

Ahuja, K., Sitaram, S., Dandapat, S., & Choudhury, M. (2022). On the calibration of massively multilingual language models. In *Proceedings of EMNLP 2022* (pp. 4310–4323). https://doi.org/10.18653/v1/2022.emnlp-main.290

Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In *Proceedings of COLING 2022* (pp. 4336–4349).

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of ICML 2017*, PMLR 70:1321–1330. arXiv:1706.04599

Islam, M. R., Seenivasan, L., Ren, H., & Glocker, B. (2021). Class-distribution-aware calibration for long-tailed recognition. arXiv:2109.05263

Muhammad, S. H., et al. (2025). AfriHate: A multilingual collection of hate speech and abusive language datasets for African languages. In *Proceedings of NAACL 2025* (pp. 1854–1871). https://doi.org/10.18653/v1/2025.naacl-long.92

---

## Appendix A — Data Card

| Field | Value |
|---|---|
| Dataset | AfriHate — Twi (`twi`) and Hausa (`hau`) subsets |
| Source | Hugging Face: `afrihate/afrihate` |
| Reference | Muhammad et al. (2025), NAACL, DOI 10.18653/v1/2025.naacl-long.92 |
| Licence | Tagged Apache 2.0 on Hugging Face; handled conservatively (see below) |
| Access method | Public dataset accessed via `datasets` library; gated — HF token required |
| Unit of analysis | Post (tweet) |
| Label strings (verified) | `Abuse`, `Hate`, `Normal` (derived from `.unique()`, not hardcoded) |
| Twi test set | 698 instances: Abuse=492, Hate=100, Normal=106 |
| Hausa test set | 1,049 instances: Abuse=351, Hate=103, Normal=595 |
| Raw data in repository | **No.** Derived outputs only: metrics, plots, aggregate confusion counts. No raw tweet text is committed, printed in notebook outputs, or included in any figure. |
| Re-identification risk | Twitter-derived text may carry residual identifiers. No re-identification, user-level linkage, or account tracing was performed. |
| Provenance note | Keyword-and-handle sampled; class balance reflects sampling strategy, not true prevalence of hate in Twi or Hausa discourse. Do not infer base rates from class counts. |
| Conservative licence handling | The Apache 2.0 tag on Hugging Face is not corroborated by a licence file in the source repository or the paper. Out of caution, no raw data is redistributed and the dataset is attributed to Muhammad et al. (2025) throughout. |

---

## Appendix B — Model Card

| Field | Value |
|---|---|
| Model type | TF-IDF (word 1–2 grams + char 2–4 grams) + Logistic Regression |
| Training data | AfriHate train split (Twi: 2,564; Hausa: 4,566 instances) |
| Validation data | AfriHate validation split (Twi: 639; Hausa: 1,029 instances) |
| Test data | AfriHate test split — held out; used only for final evaluation |
| Class weighting | `balanced` (inverse frequency) |
| Seed | 42 (deterministic; results fully reproducible) |
| Intended use | **Diagnostic instrument for calibration research only.** |
| **NOT suitable for** | **Deployment in any live moderation system.** |
| Performance (Twi) | Macro-F1 = 0.594; Abuse F1 = 0.92, Hate F1 = 0.62, Normal F1 = 0.24 |
| Performance (Hausa) | Macro-F1 = 0.738; Abuse F1 = 0.74, Hate F1 = 0.65, Normal F1 = 0.83 |
| Calibration (Twi) | Macro-avg ECE = 0.077 (before scaling); 0.065 (after) |
| Calibration (Hausa) | Macro-avg ECE = 0.038 (before scaling); 0.023 (after) |
| Known failure modes | (1) Twi Normal class: very low recall (0.14), 91 test errors, 36 at high confidence. (2) Temperature scaling worsens Twi Normal ECE by 42%. (3) High blinding rate for Twi Abuse errors (45.5%). |
| Risk statement | These classifiers are weak baselines built to investigate calibration, not to make moderation decisions. Deploying either model — with or without temperature scaling — in a content moderation pipeline without substantial additional validation would be irresponsible. The Twi Normal class in particular is badly handled, and the model would over-censor ordinary Twi speech at high confidence. |
| AI assistance disclosure | Code scaffolding and report writing assistance was provided by Claude (Anthropic). All results were verified against notebook outputs. No AI tool was used to generate data, labels, or citations. |
