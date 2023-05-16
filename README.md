# Development and Verification of a Time Series AI Model for Liver Injury Detection Based on a Multicenter Distributed Research Network

## Purpose
The purpose of this project is to develop and validate a Drug-Induced Liver Injury (DILI) prediction model using the Common Data Model (CDM). The model is designed to detect DILI using time series data from a multicenter distributed research network.

## Description
This repository contains the following scripts:

### scripts
- 1_create_cohort_person_in_db.ipynb: Creates a cohort of patients with DILI and non-DILI.
- 2_propensity_score_matching.ipynb: Matches DILI and non-DILI patients using propensity score matching.
- 3_merge_domain_data.ipynb: combines domain data in the CDM.
- 4_feature_selections.ipynb: Performs feature selection for the model.
- 7_preprocessing_lstm.ipynb: Preprocesses data for the time series model.
- 8_imv_lstm_attention.ipynb: Trains and evaluates the time series model with attention mechanism.

### scripts2
- for extract demographic data and analysis result

### Getting Started

To get started with the project, follow these steps:

1. Install project-related requirements in Python (if necessary, create a virtual environment) by running the following command:
bash
'''
pip install -r requirements.txt
pip install psycopg2-binary
'''
2. Edit the config.json file with the appropriate parameters:
- working_date: Date to run the program.
- dbms: MSSQL or PostgreSQL.
- mssql or postgresql: Server, user, password, and port.
- meas: meas_concept_id (used by the institution).
- translatequerysql: CDM database schema, target database schema, and target database schema.

3. Execute the Jupyter Notebook files in the scripts folder in the order of folder number.

### Results Export
The following directories are generated during the program execution:

- data: Data for model training.
- result: Preprocessing results and evaluation of learning performance.
To export the results, compress and export the result directory.

## Help
If you need help with the project, please refer to the documentation or contact the authors.
- suncheolheo, hepsie50@gmail.com

## Authors
This project was developed by a researcher at the DHLab, Department of Biomedical Systems Informatics, Yonsei University College of Medicine, Seoul.

## Version History
- v1.0.0 : Initial Release

## Acknowledgments
This study was supported by a grant from the Korea Institute of Drug Safety and Risk Management in 2021.

## Citation
preprint (in preparation)