# Code and Data for  
## “Unraveling the mechanism behind the probable extinction of the B/Yamagata lineage of influenza B viruses”

This repository provides the code and datasets necessary to reproduce the analyses and results reported in the article published in Nature Communications (2025):
https://doi.org/10.1038/s41467-025-65396-6
  
---

## Repository Structure

### `./code`
- **PREDAC/**
  - `BV/`: Python scripts for constructing antigenic prediction models of *B/Victoria*  
  - `BY/`: Python scripts for constructing antigenic prediction models of *B/Yamagata*  

- **dynamic_model/**
  - `parameters_estimation/`: Scripts for parameter estimation using MCMC  
  - `simulation/`: Scripts for scenario analyses  
  - `sensitivity_analysis/`: Scripts for sensitivity analyses of key parameters  

- **figure/**
  - Scripts for generating the figures included in the manuscript  

> Files with the suffix `_command` are executable scripts for running the corresponding analyses.

---

### `./data`
Contains datasets required by the scripts in `./code/` as well as the output results.

---
