# CNN on Rendered Sample Point Clouds vs. ELA+RF for Black-Box Landscape Classification

This project investigates whether a CNN operating directly on rendered images of black-box sample point clouds can replace hand-crafted ELA features for algorithm selection on the 24 BBOB benchmark functions.

## Overview

In black-box optimization, we only have access to sampled input-output pairs — no gradients, no formulas. The standard way to characterize these landscapes is Exploratory Landscape Analysis (ELA), which computes aggregate statistical features but discards all spatial and relational structure between sample points.

We render 500 sample evaluations from each BBOB function (5D, 10 instances each) as color-coded scatter images in two formats:

- **Type A (PCA scatter):** 128×128 image, points projected onto first two principal components, color encodes rank-normalized fitness
- **Type B (Pairwise grid):** 256×192 image, scatter panels for each pair of input dimensions

Both CNN models use a frozen ResNet-18 backbone with a trained classification head, class-weighted cross-entropy loss, and data augmentation. Evaluated under leave-one-problem-out (LOPO) cross-validation against an ELA+Random Forest baseline on the same sample budget.

## Results

| Method | Accuracy | Macro F1 |
|---|---|---|
| CNN (Type A, PCA) | 0.408 ± 0.259 | 0.200 ± 0.125 |
| CNN (Type B, Pairwise) | 0.442 ± 0.327 | 0.249 ± 0.167 |
| ELA + RF | 0.650 ± 0.425 | 0.559 ± 0.439 |

The CNN models did not outperform ELA+RF. Grad-CAM analysis showed the CNN does not consistently attend to informative regions of the point cloud images, and a budget sweep revealed both methods peak at N=200 samples.

## Notebook

Full code for data generation, image rendering, model training, evaluation, Grad-CAM, and budget analysis:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kGfTn0Dnmp_DE081mXl5ETO8q36ITciH?usp=sharing)

## Paper

The project paper (`paper.tex`) uses the Springer Nature sn-jnl template and covers introduction, literature review, methodology, and full results with confusion matrices, Grad-CAM interpretability, and sample budget comparison.

## References

- Nikolikj, Doerr, Eftimov — *RF+clust for Leave-One-Problem-Out Performance Prediction* (EvoApplications 2023)
- Kostovska et al. — *The Importance of Landscape Features for Performance Prediction of Modular CMA-ES Variants* (GECCO 2022)
- Cenikj, Petelin, Eftimov — *ClustOpt: Clustering-based Approach for Representing Search Dynamics* (CEC 2025)
- Mersmann et al. — *Exploratory Landscape Analysis* (GECCO 2011)
- Kerschke et al. — *Automated Algorithm Selection: Survey and Perspectives* (Evolutionary Computation, 2019)
