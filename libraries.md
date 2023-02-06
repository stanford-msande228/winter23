---
layout: page
title: Python Libraries for CausalML
description: A collection of python libraries that are useful for causal machine learning.
---

A collection of python libraries that are useful for causal machine learning.

# General
- [PyWhy](https://www.pywhy.org/). an organization that holds a multitude of causal machine learning packages in python
- [DoWhy](https://www.pywhy.org/dowhy/v0.9.1/). A package that offers a high level API for graph definition, identification, estimation, sensitivity analysis and refutation
- [PGMPY](https://pgmpy.org/index.html). A package for general Bayesian network definition and inference (not just causal inference)

# Identification
- [Ananke](https://ananke.readthedocs.io/en/latest/notebooks/quickstart.html). A package that implements advanced graph based identification algorithms and allows for semi-parametric estimation.
- [y0](https://github.com/y0-causal-inference/y0). Advanced graph based identification algorithms for general causal graphs (even allowing for un-directed edges)

# Estimation
- [EconML](https://econml.azurewebsites.net/). Many causal ML algorithms for estimation and confidence intervals for heterogeneous treatment effects (under conditional exogeneity or with instruments)
- [CausalML](https://causalml.readthedocs.io/en/latest/about.html). Many causal ml algorithms for estimation of heterogeneous treatment effects
- [UpliftML](https://upliftml.readthedocs.io/en/latest/index.html). A smaller set of causal ML algorithms for heterogeneous effect estimation, but which scales on Spark, using PySpark
- [DoubleML](https://docs.doubleml.org/stable/api/api.html). Implements the double ML estimation algorithm with inference, for average treatment effects under exogeneity or with instruments

# Causal Graph Discovery
- [PyWhy-Graphs](https://github.com/py-why/pywhy-graphs). A basic library for causal graph manipulation and basic graph algorithms (e.g. conditional independence)
- [DoDiscover](https://github.com/py-why/dodiscover). A package for causal discovery in python
- [Causica](https://github.com/microsoft/causica). A deep learning based causal discovery package
- [CausalLearn](https://github.com/py-why/causal-learn). Causal discovery package with many recent advanced algorithms from the research community