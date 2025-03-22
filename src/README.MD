# Replicating the Experiments

This repository contains scripts to process and model ECG data for predictive tasks. Follow the steps below to replicate the experiments.

## 1. Download Required Datasets
Before running the scripts, download the following files and place them in the root directory of this repository:

- **records_w_diag_icd10.csv** from [MIMIC-IV-ECG-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)
- **machine_measurements.csv** from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **ECG_ViEW_II_for_CVS.zip** (do not unzip) from [ECG ViEW II](http://ecgview.org/)

## 2. Run Processing and Modelling Script
To process the data and train models, run the following command:

```bash
python processing_and_modelling.py
```

## 3. Output and Results
The script will generate performance figures and Shapley values, which will be saved automatically in the `figures/` directory.

If you encounter any issues, check your dataset paths and ensure dependencies are correctly installed.

---

For further inquiries, please open an issue.
