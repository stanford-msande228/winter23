---
layout: home
title: Syllabus
nav_exclude: false
permalink: /:path/
seo:
  type: Course
  name: MS&E 228 – Applied Causal Inference Powered by ML and AI
---

# Welcome to MS&E 228 – Applied Causal Inference Powered by ML and AI

Instructor: Vasilis Syrgkanis  
Units: 3  
Winter Quarter 2023  
Tue, Thu 1:30-2:50PM McCullough 115

**Description:**  
The course will cover fundamentals of modern applied causal inference. Basic principles of causal inference and machine learning and how the two can be combined in practice to deliver causal insights and policy implications in real world datasets, allowing for high-dimensionality and flexible estimation. Lectures will provide foundations of these new methodologies and the course assignments will involve real world data and synthetic data analysis based on these methodologies. 

**Prerequisites:** Basic knowledge of probability and statistics. Recommended: 226 or equivalent.

### Office Hours:  (Starting Week 2)

|                   | Time                       | Location  |
|-------------------|----------------------------|-----------|
| Vasilis Syrgkanis | Thursday 3-4pm             | Huang 252 |
| Johannes Ferstad  | Tuesdays 3-4pm             | TBA       |
| Hui Lan           | Wednesdays 9:30 - 10:30 am | TBA       |

 

### Format:

The course will consist of lectures and students-led discussions. The lectures will cover fundamentals that fuse classical structural equation models (SEMs) and DAGs, with tools for statistical inference based on machine learning  (lasso, random forest, deep neural networks)  to infer causal parameters and quantify uncertainty. Grading will be primarily based on the weekly homework assignments and secondarily on class participation. There will be a total of 7-8 homeworks, rolled out roughly on a weekly basis, that will involve either mathematical proofs or coding exercises.

 

### Grading:

* Homework 90%
* Participation 10%

 

# Course Plan:

**Lecture 1:** Introduction; case studies; importance of causality; importance of handling high dimensional data/flexible modeling;

 

### Experiments and causality

**Lecture 2:** Causality via Experiments; Potential Outcomes framework; Two means estimate and confidence interval/asymptotic distribution; limitations of trials; what if we have pre-treatment co-variates: precision and heterogeneity

 
### Inference with linear models

**Lecture 3:** Basics of statistical inference in linear models; confidence intervals for p << n; simultaneous confidence bands; interpretation of coefficient as partialling out; inference on ATE from trials via regression; Revisiting the role of covariates in randomized trials: precision and heterogeneity: variance characterization and comparisons

**Lecture 4:** High dimensional methods and prediction; regularization; lasso; elasticnet;

**Lecture 5:** Inference in high-dimensional methods; double lasso; partialling out; intro to Neyman orthogonality

 
### Observational data, causality, DAGs

**Lecture 6:** Causality in observational data; confounding; conditional ignorability;  identification by conditioning; identification via propensity scores

**Lecture 7:** Structural equations models and DAGs; basics of DAGs; conditional ignorability in DAGs; Good and Bad controls

**Lecture 8:** General DAGs and Counterfactuals; SWIGs; D-separation; Interventions; Re-visting identification by conditioning

 

### ML estimation of non-linear models

**Lecture 9:** Modern methods for non-linear prediction: trees and forests; neural networks; feature engineering; some guarantees

**Lecture 10:** Ensembling; stacking; auto-ML

 

### Statistical inference with non-linear models

**Lecture 11:** DML for PLR and fully non-linear for ATE; Generic debiased ML framework

 

### Causal Discovery

**Lecture 12:** Causal discovery

 

### Unobserved Confounding

**Lecture 13:** Omitted variable bias;  Instrumental variables; LATE

**Lecture 14:** Inference in PL IV and non-linear IV models; inference with weak instruments; DML with weak identification

 

### Heterogeneous Effects and Policy Learning

**Lecture 15:** CATE methods; meta learners; neural network methods; policy learning

**Lecture 16:** Evaluation and model selection of CATE methods; for Trials; for Observational Data


### Further Topics (Subject to change)

**Lecture 17:** Censoring

**Lecture 18:** Dynamic regime; Optimal regime; off-policy RL; Surrogates

**Lecture 19:** More structural approaches to un-observed confounding: diff-in-diff; synthetic controls; regression discontinuity (soft RD); proximal inference

**Lecture 20:** Helicopter tour of current software ecosystem for causal machine learning; A helicopter tour of what we did not cover in the course: active experiments