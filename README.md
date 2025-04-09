# GECCO 2025 - Evaluating the Generalizability of Machine Learning Pipelines When Using Lexicase or Tournament Selection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15171303.svg)](https://doi.org/10.5281/zenodo.15171303)
[![supplemental](https://img.shields.io/badge/go_to-supplementary_material-98111e)](https://jgh9094.github.io/lexidate-holdout-analysis/Supplementary-Material/)
[![data](https://img.shields.io/badge/go_to-data-9807FF)](https://osf.io/g5u9p/)

## Abstract

> Evolutionary algorithms have been successfully combined with Automated Machine Learning (AutoML) to evolve effective machine learning pipelines.
Here, we use 12 OpenML classification tasks and the AutoML tool TPOT2 to assess the impact of lexicase and tournament selection on the generalizability of pipelines.
We use one of five stratified sampling splits to generate training and validation sets; pipelines are trained on the training set, and predictions are made on the validation set.
Lexicase and tournament selection use these predictions to identify parents.
At the end of a run, TPOT2 returns the pipeline that achieved the best validation accuracy while maintaining the lowest complexity.
The generalizability of this pipeline is assessed using the test set provided for an OpenML task.
We found that lexicase produced pipelines with higher validation accuracy than tournament selection in all tasks for at least one split.
In contrast, tournament selection produced pipelines with greater generalizability for 10 of the 12 tasks on at least one split.
For most cases where tournament selection outperformed lexicase on test accuracy, we detected differences in validation accuracy and pipeline complexity.

## Repository guide

Datasets used in the experiments. The `Task ID' refers to the identifier used to extract the dataset from OpenML. The other columns denote the number of rows, columns, and classes for each dataset.

- `Data-Tools/`: all scripts related to data checking, collecting, and visualizing
  - `Check/`: scripts for checking data
  - `Collector/`: scripts for collecting data
  - `Task-List/`: details on training set for each OpenML task used
  - `Visualize/`: scripts for making plots
- `Experiments/`: all scripts to run experiments on HPC
  - `Lexicase/`: runs related to lexicase selection
  - `Tournament/`: runs related to tournament selection
- `Source/`: contains all Python scripts to run experiments.
- `3-step-ea`: TPOT2 implementation for this work: evaluation -> selection -> reproduction


## OpenML classification tasks

| Name                    | Task ID | Rows | Columns | Classes |
|-------------------------|---------|------|---------|---------|
| australian              | 146818  | 621  | 15      | 2       |
| eucalyptus              | 359954  | 662  | 91      | 5       |
| blood-transfusion...    | 359955  | 673  | 4       | 2       |
| vehicle                 | 190146  | 761  | 18      | 4       |
| credit-g                | 168757  | 900  | 61      | 2       |
| qsar-biodeg             | 359956  | 949  | 41      | 2       |
| pc4                     | 359958  | 1312 | 37      | 2       |
| cmc                     | 359959  | 1325 | 24      | 3       |
| yeast                   | 2073    | 1335 | 8       | 10      |
| car                     | 359960  | 1555 | 21      | 4       |
| steel-plates-fault      | 168784  | 1746 | 27      | 7       |
| kc1                     | 359962  | 1898 | 21      | 2       |

## TPOT2 configurations

| Parameter                | Values        |
|--------------------------|---------------|
| Population size          | 100           |
| Number of generations    | 200           |
| Mutation rate            | 70%           |
| Crossover rate           | 30%           |
| Number of runs per condition | 40        |

## How to run

Please refer to 3-step-ea/README.md for instructions on how to install tpot2.
Once all installation is complete, verify that the conda enviornment matches the one used in this work to run jobs.
Here, we expect 'conda activate tpot2-env-3.10' to run these jobs.