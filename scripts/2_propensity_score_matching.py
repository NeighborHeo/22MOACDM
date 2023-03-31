# %%
"""
## 1. PS(Propensity score) Matching 
  * 1) Retrieve patient information from DB
  * 2) PS Matched, and saved patients data in the local folder.
"""

# %%
# In[ ]:
"""
1) import package
"""
import os
import sys
import json
import pathlib
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
from datetime import timedelta

from _utils.customlogger import customlogger as CL
from _utils.preprocessing import *
from _utils.psmatch import *

pd.set_option('display.max_colwidth', -1)  #각 컬럼 width 최대로 
pd.set_option('display.max_rows', 50)      # display 50개 까지 

# %%
# In[ ]:
"""
2) loading config
"""
current_dir = pathlib.Path.cwd()
parent_dir = current_dir.parent
with open(parent_dir.joinpath("config.json")) as file:
    cfg = json.load(file)

# %%
# In[ ]:
"""
3) load information 
"""
working_date = cfg["working_date"]
curr_file_name = os.path.splitext(os.path.basename(os.path.abspath('')))[0]

# %%
# In[ ]:
"""
4) create Logger
"""
log = CL("custom_logger")
pathlib.Path.mkdir(pathlib.Path('{}/_log/'.format(parent_dir)), mode=0o777, parents=True, exist_ok=True)
log = log.create_logger(file_name="../_log/{}.log".format(curr_file_name), mode="a", level="DEBUG")  
log.debug('start {}'.format(curr_file_name))

# %%
# # In[ ]:
# """
# 5) connection DataBase
# """
# if (cfg["dbms"]=="postgresql"):
#     db_cfg = cfg["postgresql"]
#     import psycopg2 as pg
#     conn = pg.connect(host=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], dbname=db_cfg['@database']) 
#     log.debug("postgresql connect")
    
# elif (cfg["dbms"]=="mssql"):
#     db_cfg = cfg["mssql"]
#     import pymssql
#     conn= pymssql.connect(server=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], database=db_cfg['@database'], as_dict=False)
#     log.debug("mssql connect")
    
# else:
#     log.warning("set config.json - sql - dbms : mssql or postgresql")

# %%
"""
5) connection DataBase
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

driver = cfg["dbms"]
db_cfg = cfg[driver]
username = db_cfg["@user"]
password = db_cfg["@password"]
host = db_cfg["@server"]
port = db_cfg["@port"]
database = db_cfg["@database"]
if cfg["dbms"] == "mssql":
    sqldriver = "mssql+pymssql"
elif cfg["dbms"] == "postgresql":
    sqldriver = "postgresql+psycopg2"
url = f"{sqldriver}://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(url, echo=True)
sessionlocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)

# %%
def runTask(outcome_name):
    """
        Propensity Score Matching
    """
    output_data_dir = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, working_date, outcome_name))
    output_result_dir = pathlib.Path('{}/result/{}/importsql/{}/'.format(parent_dir, working_date, outcome_name))
    pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
    pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)


    # In[ ]:
    """
        (0) Only the measurement concept IDs used to extract side effects are extracted. To match PS (Propensity Score)
    """
    outcome_lab_list = []
    if 'nephrotoxicity' == cfg['drug'][outcome_name]['ade'] :
        concept_id_CR = int(cfg['meas']["CR"]['@meas_concept_id'])
        outcome_lab_list = [concept_id_CR]
    else :
        concept_id_AST = int(cfg['meas']["AST"]['@meas_concept_id'])
        concept_id_ALT = int(cfg['meas']["ALT"]['@meas_concept_id'])
        concept_id_ALP = int(cfg['meas']["ALP"]['@meas_concept_id'])
        concept_id_TB = int(cfg['meas']["TB"]['@meas_concept_id'])
        outcome_lab_list = [concept_id_AST, concept_id_ALT, concept_id_ALP, concept_id_TB]
    outcome_lab_list_str = ','.join([str(i) for i in outcome_lab_list])
    print(outcome_lab_list_str)

    # In[ ]:
    """
        (1) set table name and Execute a query to read a table
    """
    tnPopulation = '{}.person_{}'.format(db_cfg["@person_database_schema"], outcome_name)
    tnMeasurement = '{}.measurement'.format(db_cfg["@cdm_database_schema"])

    sql_person_query = f"""
    SELECT * FROM {tnPopulation}
    """
    
    sql_meas_query = f"""
    SELECT person_id, measurement_concept_id, measurement_date, value_as_number, range_low, range_high 
    FROM {tnMeasurement}
    WHERE measurement_concept_id in ({outcome_lab_list_str})
    AND value_as_number IS NOT NULL
    AND person_id IN (SELECT person_id FROM {tnPopulation});
    """

    person_df = pd.read_sql(sql=sql_person_query, con=engine)
    meas_df = pd.read_sql(sql=sql_meas_query, con=engine)

    # In[ ]:
    """
        (2) Convert gender to sex columns to 0,1
            : Change the column name to sex and display 0 for female and 1 for male
    """
    def convert_gender_column(_df, inplace=False):
        df = _df if inplace==True else _df.copy()
        
        if 'gender_source_concept_id' in df.columns and df['gender_source_concept_id'].notnull().all():
            print("selected gender_source_concept_id")
            df.rename(columns={'gender_source_concept_id':'sex'}, inplace=True)
            df['sex'].replace(8532, 0, inplace=True)
            df['sex'].replace(8507, 1, inplace=True)
        elif 'gender_source_value' in df.columns and df['gender_source_value'].notnull().all():
            print("selected gender_source_value")
            df.rename(columns={'gender_source_value':'sex'}, inplace=True)
            df['sex'].replace(['F', 'Female'], 0, inplace=True)
            df['sex'].replace(['M', 'Male'], 1, inplace=True)
        else :
            print("The gender column has already been changed, there is no column, or the data is null.")
        return df
    
    person_df = convert_gender_column(person_df)
    
    # In[ ]:
    """
        (4) Label 1 if the first abnormal date value exists, 0 if not.
    """
    person_df['label'] = (~person_df['first_abnormal_date'].isnull()).astype(int)

    # concat할 수 있도록 column name 통일시켜주기
    meas_df.rename(columns={'measurement_concept_id':'concept_id','measurement_date':'concept_date','value_as_number':'concept_value'}, inplace=True)

    # In[ ]:
    """
        (6) Remove duplicate data
    """
    def drop_duplicates_(domain_df):
        n_prev = len(domain_df)
        domain_df = domain_df.drop_duplicates()
        n_next = len(domain_df)
        print('{}>{}'.format(n_prev, n_next))
        return domain_df
    meas_df = drop_duplicates_(meas_df)

    # In[ ]:
    """
        (7) person(+cohort) + measurement table
    """
    meas_df = pd.merge(person_df, meas_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
    meas_df = cohortConditionSetting(meas_df, pre_observation_period=60, post_observation_peroid=60, delDataAfterAbnormal=True)
    psm_data_df = meas_df.loc[meas_df['concept_id'].isin(outcome_lab_list)]
    
    # In[ ]:
    """
        (9) Pivoting Lab Data Rows to Columns
    """
    psm_data_df = psm_data_df.query('concept_date <= cohort_start_date')
    psm_data_df = pd.pivot_table(data=psm_data_df, index=['person_id', 'age', 'sex', 'label'], columns='concept_id', values='concept_value').reset_index().rename_axis(None, axis=1)
    psm_data_df.columns = psm_data_df.columns.astype(str)
    print(psm_data_df.head(), psm_data_df.dtypes)

    # In[ ]:
    """
        (10) PS (Propensity Score) Matching with KNN Match algorithm using psmpy library
    """
    psm_data_df = psm_data_df.dropna()
    # covarates = columns - person_id, label 
    covarates = list(set(psm_data_df.columns) - {'person_id', 'label'})
    matched_df = get_matching_multiple_pairs(psm_data_df, treatments='label', covariates=covarates, n_neighbors=3)

    # In[ ]:
    """
        (11) Filtered by PSMatched patient IDs and saved as a file
    """
    psm_person_ids = matched_df.person_id.values
    person_df = pd.read_sql(sql=sql_person_query, con=engine)
    psm_person_df = person_df.loc[person_df.person_id.isin(psm_person_ids)].reset_index(drop=True)
    psm_person_df.to_sql(name=f"person_{outcome_name.lower()}_psm", schema=cfg[driver]["@person_database_schema"], con=engine, if_exists='replace', index=False)
    psm_person_df.to_csv(output_data_dir.joinpath('psm_person_df.txt'),index=False)
    import pickle
    with open(output_data_dir.joinpath('psm_person_ids.pkl'), 'wb') as f:
        pickle.dump(psm_person_ids, f)
        
    for cov in covarates:
        print(f'covariate: {cov}')
        p_value = user_t_test_ind(matched_df, 'label', cov)
        print("cov : {} / p value: {}".format(cov, p_value))


# %%
# In[ ]:
"""
    For all drugs, perform the above tasks.
"""
for outcome_name in tqdm(cfg['drug'].keys()) :
    try :
        log.debug('drug : {}'.format(outcome_name))
        runTask(outcome_name)
    except :
        traceback.print_exc()
        log.error(traceback.format_exc())


# %%
engine.dispose()