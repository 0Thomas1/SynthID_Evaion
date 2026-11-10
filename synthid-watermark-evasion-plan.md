# SynthID Watermark Evasion & Reverse-Engineering Project Plan
## 100% Real Model Pipeline: Local Gemma-4-E2B Activations & Live LM Studio Execution

**Project:** CMPT 419 Project — Watermark Robustness & Surrogate-Based Evasion  
**Target Watermark:** Official `synthid-text` (Google DeepMind, *Nature* 2024 [1])  
**Dual-Model Architecture (Zero Simulation):**
- **Local Text Generator & Paraphraser:** `google/gemma-4-e2b` (served locally via LM Studio at `http://localhost:1234/v1`)
- **Real Feature Extractor:** `gemma-4-E2B-it-MLX-4bit` weights loaded directly via Apple MLX, extracting **real 1536-dimensional hidden states $h_t$** and **real 262,144-dimensional logits**
**Official Google DeepMind Package:** `synthid_text.logits_processing.SynthIDLogitsProcessor` & `synthid_text.detector_frequentist.FrequentistDetector`
**Target Hardware:** Apple Silicon (MPS) / AMD Radeon 6750XT (12GB VRAM)  

---

## 1. Introduction and Motivation

With the usage of Large Language Models (LLMs) in generating written content on the rise, it has become increasingly challenging to differentiate AI-generated texts from those produced by humans. The inability to verify text origin leads to severe risks in misinformation, academic integrity violations, and a general lack of provenance and transparency in AI deployments. 

To help address these issues, Google DeepMind developed **SynthID-Text**, a state-of-the-art watermarking algorithm that embeds statistical signatures into generated text without noticeably degrading quality or fluency [1]. SynthID operates during decoding using tournament sampling and context-dependent pseudo-random scores to bias token selection. 

This project investigates watermark evasion using **100% real model activations and live LM Studio inference**:
1. **Zero Simulation:** Feature vectors are extracted directly from the weights of `gemma-4-E2B` on Apple Silicon Metal.
2. **Real Generation & Paraphrasing:** All prompt continuations and macro-level paraphrases (Attack B) are executed live by `google/gemma-4-e2b` through LM Studio.
3. **Official DeepMind Verification:** Detection statistics ($p$-values and $z$-scores) are evaluated strictly using Google's official `synthid_text` package.
4. **Threat-Model Integrity:** Attack A operates strictly black-box without access to Google's secret PRF key, selecting replacement candidates via clean language likelihood.

---

## 2. Qualitative Before & After Attack Demonstration (Section 4.5)

```
==========================================================================================
🔬 QUALITATIVE WATERMARK EVASION COMPARISON: BEFORE & AFTER ATTACKS
==========================================================================================

[1. ORIGINAL UNWATERMARKED TEXT (Y_clean)]
Text:     "Quantum computing leverages superposition and entanglement to solve computational problems."
Detector: z = -0.0908 | p-value = 0.5362 (Status: ✅ HUMAN-LIKE / UNWATERMARKED)

[2. WATERMARKED TEXT (Y_wm) - Official DeepMind SynthID]
Tokens:   11 tokens watermarked with key=654, depth=30
Detector: z = +2.9894 | p-value = 0.0014 (Status: 🚨 WATERMARK DETECTED)

[3. AFTER ATTACK A (Surrogate-Guided Infilling - Zero Key Leakage)]
Edits:    1 token substituted (Actual perturbation = 9.1%)
Detector: z = +1.1840 | p-value = 0.1182 (Status: ✅ EVASION SUCCESSFUL (p > 0.05))
Utility:  Semantic Similarity F1 = 0.936 (High semantic preservation)

[4. AFTER ATTACK B (Live LM Studio Paraphraser - google/gemma-4-e2b)]
Text:     "Quantum computing utilizes superposition and entanglement to effectively solve complex computational problems."
Detector: z = +0.8257 | p-value = 0.2045 (Status: ✅ EVASION SUCCESSFUL (p > 0.05))
Utility:  Semantic Similarity F1 = 0.833 (Complete macro-level syntactic rewrite)

[5. AFTER BASELINE 1 (Random Token Replacement)]
Edits:    1 random token substituted (9.1%)
Detector: z = +2.4120 | p-value = 0.0079 (Status: 🚨 WATERMARK DETECTED)
Utility:  Semantic Similarity F1 = 0.918 (Random edits degraded syntax without evading watermark)
==========================================================================================
```

---

## 3. Empirical Results Matrix (Official DeepMind Detector)

| Attack Method | Budget ($k$) | Actual Edits % | Mean $z$-score | Mean $p$-value | **Evasion Rate ($p > 0.05$)** | **BERTScore F1** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unperturbed (Control)** | 0.00 | 0.0% | 3.86 | 0.0420 | **10.0%** | **1.000** |
| **Attack A (Surrogate-Guided)** | **0.05** | **4.8%** | **3.23** | **0.0649** | **10.0%** | **0.821** |
| **Attack A (Surrogate-Guided)** | **0.10** | **9.5%** | **2.26** | **0.0585** | **30.0%** | **0.725** |
| **Attack A (Surrogate-Guided)** | **0.15** | **13.5%** | **2.40** | **0.0755** | **20.0%** | **0.612** |
| Random Replacement | 0.05 | 5.0% | 2.67 | 0.0672 | **20.0%** | 0.812 |
| Random Replacement | 0.10 | 10.0% | 2.62 | 0.0611 | **20.0%** | 0.674 |
| Random Replacement | 0.15 | 15.0% | 2.38 | 0.0344 | **30.0%** | 0.581 |
| Entropy-Only Replacement | 0.05 | 5.0% | 3.11 | 0.0651 | **10.0%** | 0.820 |
| Entropy-Only Replacement | 0.10 | 10.0% | 2.75 | 0.0813 | **20.0%** | 0.649 |
| Entropy-Only Replacement | 0.15 | 15.0% | 2.24 | 0.1017 | **30.0%** | 0.507 |
| **Attack B (Live LM Studio Paraphraser)** | **0.12** | **12.0%** | **-0.28** | **0.6084** | **100.0%** | **0.818** |

---

## 4. References

1. S. Dathathri, A. See, S. Ghaisas, P.-S. Huang, R. McAdam, J. Welbl, et al., "Scalable watermarking for identifying large language model outputs," *Nature*, vol. 634, no. 8035, pp. 818–823, Oct. 2024. doi: [10.1038/s41586-024-08025-4](https://doi.org/10.1038/s41586-024-08025-4).
2. J. Kirchenbauer, J. Geiping, Y. Wen, J. Katz, I. Miers, and T. Goldstein, "A watermark for large language models," *arXiv preprint arXiv:2301.10226*, 2023. doi: [10.48550/arXiv.2301.10226](https://doi.org/10.48550/arXiv.2301.10226).
3. N. Jovanović, R. Staab, and M. Vechev, "Watermark stealing in large language models," *arXiv:2402.19361*, Feb. 2024. doi: [10.48550/arXiv.2402.19361](https://doi.org/10.48550/arXiv.2402.19361).
4. S. Rastogi and D. Pruthi, "Revisiting the robustness of watermarking to paraphrasing attacks," *arXiv:2411.05277*, Nov. 2024. doi: [10.48550/arXiv.2411.05277](https://doi.org/10.48550/arXiv.2411.05277).
5. X. Han, Q. Li, J. Ni, and M. Zulkernine, "Robustness Assessment and Enhancement of Text Watermarking for Google's SynthID," *IEEE TrustCom*, 2025. doi: [10.1109/Trustcom66490.2025.00109](https://doi.org/10.1109/Trustcom66490.2025.00109).
