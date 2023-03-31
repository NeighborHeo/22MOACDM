# MOA CDM-based drug side effect artificial intelligence model development and validation​

Purpose​ : Development and validation of drug-induced nephrotoxicity and hepatotoxicity prediction model using Common Data Model

## Description
├── scripts  
│   ├── 1_create_cohort_person_in_db.ipynb  
│   ├── 2_propensity_score_matching.ipynb  
│   ├── 3_merge_domain_data.ipynb  
│   ├── 4_feature_selections.ipynb  
│   ├── 5_preprocessing_xgboost.ipynb  
│   ├── 6_xgboost.ipynb  
│   ├── 7_preprocessing_lstm.ipynb  
│   ├── 8_imv_lstm_attention.ipynb  

### Installing

Install project-related requirements in Python
(If necessary, create a virtual environment)

pip install -r requirements.txt

and

~~graphviz install (https://graphviz.org/download/)~~
- Check installation
  : cmd or terminal > "dot -V"

and 

pip install psycopg2-binary

## Getting Started

edit config.json file

* 'working_date' : Date to run the program.
* 'dbms' : mssql or postgresql
* 'mssql' or 'postgresql' : server / user /password / port .. 
* 'meas' : meas_concept_id (used by the institution)
* 'translatequerysql' : cdm_database_schema / target_database_schema / target_database_schema

### Executing program

run python script
   1-1) Execute ipynb files in the order of folder number

### Result export
   - data (data for model training)
   - result (Preprocessing results / evaluation of learning performance)
   > Compress and export the result dir

## Help

-

## Authors

Researcher, DHLab, Department of Biomedical Systems Informatics,
Yonsei University College of Medicine, Seoul

## Version History

-

## License

-

## Acknowledgments

-
