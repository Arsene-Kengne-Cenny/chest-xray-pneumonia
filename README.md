# Chest X-Ray Pneumonia Detection — CNN vs Transfer Learning

## Project Overview

This project builds and evaluates a deep learning system to classify chest X-ray images as **Normal** or **Pneumonia** using a publicly available Kaggle dataset.

Pneumonia detection is a high-impact medical imaging task where **error type matters more than raw accuracy**. False negatives may delay treatment, while excessive false positives increase clinical workload. This project focuses not only on predictive performance, but on **error balance, generalization, and model behavior analysis**.

The central question:

> How does a CNN trained from scratch compare to a fine-tuned pretrained network when data is limited?

---

## Models Implemented

### 1) Custom CNN (trained from scratch)

* Convolutional Neural Network implemented in PyTorch
* Weighted loss to address class imbalance
* Regularization via weight decay (L2)
* Early stopping
* Full training from scratch on medical data

This model serves as a controlled baseline.

---

### 2) Fine-Tuned ResNet18 (Transfer Learning)

* Pretrained on ImageNet
* Last residual block + classification head fine-tuned
* Binary classification setup
* Same evaluation pipeline as baseline

This model tests how pretrained representations improve generalization and error balance.

---

## Evaluation Strategy

Models were evaluated on a held-out test set (n = 624) using:

* Test accuracy
* ROC-AUC
* Confusion matrix
* Precision / Recall / F1-score
* Balanced accuracy
* Grad-CAM interpretability analysis

The focus is on:

* Sensitivity vs specificity trade-off
* False negative vs false positive behavior
* Model calibration at threshold 0.5
* Visual attention patterns via Grad-CAM

---

# Final Results

| Model                 | Test Accuracy | ROC-AUC |  TN |  FP | FN |  TP |
| --------------------- | ------------: | ------: | --: | --: | -: | --: |
| Custom CNN (best)     |          0.68 |   0.919 |  36 | 198 |  1 | 389 |
| ResNet18 (fine-tuned) |          0.90 |   0.979 | 175 |  59 |  2 | 388 |

---

## Key Observations

### 1️⃣ Overall Generalization

The custom CNN achieves a ROC-AUC of 0.919, showing it learns meaningful representations. However, its accuracy drops to 0.68 due to an excessive number of false positives at threshold 0.5.

ResNet18 significantly improves performance with:

* 0.90 test accuracy
* 0.979 ROC-AUC
* Stronger separation between classes

Transfer learning clearly enhances robustness and generalization.

---

### 2️⃣ Error Balance (Clinical Perspective)

**Custom CNN**

* FN = 1 (very high sensitivity)
* FP = 198 (very low specificity)
* Normal recall = 0.15

The model strongly over-predicts pneumonia.

**ResNet18**

* FN = 2 (still extremely low)
* FP = 59 (drastically reduced)
* Normal recall = 0.75

ResNet18 provides a far better sensitivity–specificity trade-off.

---

### 3️⃣ ROC-AUC vs Accuracy

ROC-AUC measures ranking quality across thresholds.
Accuracy depends on a fixed threshold (0.5).

The CNN separates classes reasonably well (AUC 0.919), but its decision boundary at 0.5 produces too many false positives.

ResNet18 maintains both high AUC and strong accuracy because class separation is stronger and the default threshold is more appropriate.

---

## Interpretability (Grad-CAM)

Grad-CAM was used to analyze attention maps on false negatives and false positives.

* The CNN produced weak or non-informative activation maps.
* ResNet18 generated more structured and localized attention patterns.
* Some false positives were associated with high-contrast anatomical regions.

This analysis reinforces that pretrained backbones learn more stable visual representations.

---

## Technical Stack

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Matplotlib / Seaborn
* Google Colab (training)
* Git for version control

---

## Reproducibility

* Final models evaluated on a held-out test set (n = 624)
* Best checkpoints loaded for final evaluation
* Full training and evaluation pipeline available in `notebooks/01_training.ipynb`

---

## Takeaways

* Training from scratch on limited medical data can achieve high sensitivity but unstable decision behavior.
* Transfer learning significantly improves error balance and robustness.
* Evaluating medical models requires analyzing confusion matrices and error types — not accuracy alone.
* Model behavior matters as much as performance metrics.

---

## Author

This project is part of my transition from a PhD background in linguistics toward applied machine learning and NLP. It demonstrates practical deep learning implementation, error analysis rigor, and model evaluation maturity.


