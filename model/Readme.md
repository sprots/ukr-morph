
# Ukrainian Morpheme Segmentation Model

A CNN-based model for morpheme segmentation in Ukrainian, enhanced with Part-of-Speech (POS) tagging and Conditional Random Fields (CRF) for sequence labeling.

---

## 📌 Model Overview

- **Architecture**: CNN + CRF with POS features  
- **LSTM**: `False`  
- **CRF**: `True`  
- **POS Features**: `True`  
- **Vocabulary Size**: 37  
- **Label Set Size**: 15 (B-/I- prefixes for morpheme boundaries)  

This model performs morpheme boundary prediction (segmentation) on Ukrainian text using character-level CNNs, leveraging POS tags as additional input features and CRF for optimal sequence decoding.

---

## ⚙️ Training Configuration

Based on the training script (`train.py`), the model was trained with the following hyperparameters:

| Parameter | Value |
|--------|-------|
| **Epochs** | 75 (early stopping applied) |
| **Batch Size** | 32 |
| **Optimizer** | Adam (`lr=0.001`, `weight_decay=1e-5`) |
| **Learning Rate Scheduler** | `ReduceLROnPlateau` (patience=5, factor=0.5) |
| **Dropout** | 0.3 |
| **Embedding Dimension** | 128 |
| **CNN Filters** | `[256, 192, 128]` |
| **Kernel Sizes** | `[5, 5, 5]` |
| **LSTM Hidden Size** | 256 (bidirectional, not used in this config) |
| **Gradient Clipping** | `max_norm=1.0` |
| **Early Stopping Patience** | 10 epochs |
| **POS Embedding Size** | 32 |

Additional linguistic features used:
- **Ukrainian Vowels**: Explicit binary feature for vowels: `а, е, є, и, і, ї, о, у, ю, я`
- **Character Frequency Ordering**: Characters mapped by frequency-based ranking from `UKRAINIAN_LETTERS` dictionary
- **POS Tags**: 17 Ukrainian POS tags supported (e.g., `N`, `V`, `A`, `R`, etc.)

---

## 📈 Training Summary

- **Epochs Trained**: 54 (early stopping triggered)
- **Final Dev F1**: **0.9762**
- **Training Loss**: Decreased from 2.4407 to 0.5182
- **Best Model Saved At**: Epoch 39 (F1 = 0.9760), continued improvement until epoch 44

Early stopping was triggered after no improvement in dev F1 for 10 consecutive epochs.

---

## 📊 Final Test Performance

| Metric       | Score |
|-------------|-------|
| **F1-Score (Weighted)** | 0.9754 |
| **Precision (Weighted)** | 0.9755 |
| **Recall (Weighted)** | 0.9755 |
| **Accuracy** | 0.98 |

### Per-Class F1 Scores
- `B-F`: 0.9943  
- `B-H`: 0.9971  
- `B-I`: 0.9604  
- `B-P`: 0.9734  
- `B-R`: 0.9704  
- `B-S`: 0.9699  
- `B-X`: 0.9989  
- `I-F`: 0.9988  
- `I-I`: 0.0000 *(very rare class, 7 instances)*  
- `I-P`: 0.9765  
- `I-R`: 0.9729  
- `I-S`: 0.9745  
- `I-X`: 0.9989  

> ⚠️ **Note**: The `I-I` class achieved F1 = 0.0 due to no predicted samples (data sparsity). Warnings from scikit-learn were suppressed accordingly.

---

## 🧰 Usage

To load and use the trained model use tools/ukr-morph-tagger.py
