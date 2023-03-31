# %%
"""
## 3. merge domain data
  * 1) Reading PS Matched IDs from a file
    2) Load patient's measurement, drug, condition, procedure information from DB 
    3) Save All domain table data to file

---------
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

from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

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
"""

"""

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
# In[ ]:
"""
    5) connection DataBase
"""
if (cfg["dbms"]=="postgresql"):
    db_cfg = cfg["postgresql"]
    import psycopg2 as pg
    conn = pg.connect(host=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], dbname=db_cfg['@database']) 
    log.debug("postgresql connect")
    
elif (cfg["dbms"]=="mssql"):
    db_cfg = cfg["mssql"]
    import pymssql
    conn= pymssql.connect(server=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], database=db_cfg['@database'], as_dict=False)
    log.debug("mssql connect")
    
else:
    log.warning("set config.json - sql - dbms : mssql or postgresql")

# %%
def runTask(outcome_name):
    """
        Make all domain tables with Measurement / Drug / Procedure / Condition tables.
    """
    output_data_dir = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, working_date, outcome_name))
    output_result_dir = pathlib.Path('{}/result/{}/importsql/{}/'.format(parent_dir, working_date, outcome_name))
    pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
    pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

    # In[ ]:
    """
        1) Read Propensity Score matching patient IDs.
    """
    import pickle
    with open(output_data_dir.joinpath('psm_person_ids.pkl'), 'rb') as f:
        psm_person_ids = pickle.load(f).tolist()
        
    # In[ ]:
    """
        2) set Table name and read table from DB
    """
    # ** set Tablename for reading from DB **
    tnPerson = '{}.person_{}_psm'.format(db_cfg["@person_database_schema"], outcome_name)
    tnMeasurement = '{}.measurement'.format(db_cfg["@cdm_database_schema"])
    tnDrug = '{}.drug_exposure'.format(db_cfg["@cdm_database_schema"])
    tnProcedure = '{}.procedure_occurrence'.format(db_cfg["@cdm_database_schema"])
    tnCondition = '{}.condition_occurrence'.format(db_cfg["@cdm_database_schema"])
    tnConcept = '{}.concept'.format(db_cfg["@cdm_database_schema"])

    sql_person_query = f"""
    SELECT * FROM {tnPerson}
    """
    sql_meas_query = f"""
    SELECT person_id, measurement_concept_id, measurement_date, value_as_number, range_low, range_high 
    FROM {tnMeasurement}
    WHERE measurement_concept_id!=0 
    AND value_as_number IS NOT NULL
    AND person_id IN (SELECT person_id FROM {tnPerson});
    """
    sql_drug_query=f"""
    SELECT person_id, drug_concept_id, drug_exposure_start_date, quantity
    FROM {tnDrug}
    WHERE drug_concept_id!=0 and quantity IS NOT NULL
    AND person_id IN (SELECT person_id FROM {tnPerson});
    """
    sql_proc_query=f"""
    SELECT person_id, procedure_concept_id, procedure_date
    FROM {tnProcedure}
    WHERE procedure_concept_id!=0
    AND person_id IN (SELECT person_id FROM {tnPerson});
    """
    sql_cond_query=f"""
    SELECT person_id, condition_concept_id, condition_start_date
    FROM {tnCondition}
    WHERE condition_concept_id!=0
    AND person_id IN (SELECT person_id FROM {tnPerson});
    """
    sql_concept_query=f"""
    SELECT concept_id, concept_name
    FROM {tnConcept}
    WHERE concept_id!=0
    AND (concept_id IN (SELECT DISTINCT measurement_concept_id FROM {tnMeasurement})
    OR concept_id IN (SELECT DISTINCT drug_concept_id FROM {tnDrug})
    OR concept_id IN (SELECT DISTINCT condition_concept_id FROM {tnCondition})
    OR concept_id IN (SELECT DISTINCT procedure_concept_id FROM {tnProcedure}));
    """
    def filterPatients(df, psm_person_ids):
        return df.loc[df.person_id.isin(psm_person_ids)].reset_index()

    person_df = pd.read_sql(sql=sql_person_query, con=conn)
    # person_df = filterPatients(person_df, psm_person_ids)
    meas_df = pd.read_sql(sql=sql_meas_query, con=conn)
    # meas_df = filterPatients(meas_df, psm_person_ids)
    drug_df = pd.read_sql(sql=sql_drug_query, con=conn)
    # drug_df = filterPatients(drug_df, psm_person_ids)
    proc_df = pd.read_sql(sql=sql_proc_query, con=conn)
    # proc_df = filterPatients(proc_df, psm_person_ids)
    cond_df = pd.read_sql(sql=sql_cond_query, con=conn)
    # cond_df = filterPatients(cond_df, psm_person_ids)
    concept_df = pd.read_sql(sql=sql_concept_query, con=conn)

    # In[ ]:
    """
        3) Temporarily save table read from DB
    """
    # ** Save dataset **
    person_df.to_csv('{}/person.txt'.format(output_data_dir),index=False)
    meas_df.to_csv('{}/measurement.txt'.format(output_data_dir),index=False)
    drug_df.to_csv('{}/drug.txt'.format(output_data_dir),index=False)
    proc_df.to_csv('{}/procedure.txt'.format(output_data_dir),index=False)
    cond_df.to_csv('{}/condition.txt'.format(output_data_dir),index=False)
    concept_df.to_csv('{}/concept.txt'.format(output_data_dir),index=False)
    log.debug('success : {}, {}, {}, {}'.format(len(meas_df), len(drug_df), len(proc_df), len(cond_df)))

    # ** Load dataset **
    # person_df=pd.read_csv('{}/person.txt'.format(output_data_dir))
    # meas_df=pd.read_csv('{}/measurement.txt'.format(output_data_dir))
    # drug_df=pd.read_csv('{}/drug.txt'.format(output_data_dir))
    # proc_df=pd.read_csv('{}/procedure.txt'.format(output_data_dir))
    # cond_df=pd.read_csv('{}/condition.txt'.format(output_data_dir))
    # concept_df=pd.read_csv('{}/concept.txt'.format(output_data_dir))

    # In[ ]:
    """
        (4) Convert gender to sex columns to 0,1
    """
    def convert_gender_column(_df, inplace=False):
        df = _df if inplace==True else _df.copy()
        """Change the column name to sex and display 0 for female and 1 for male"""
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
        (5) Label 1 if the first abnormal date value exists, 0 if not.
    """
    person_df['label'] = (~person_df['first_abnormal_date'].isnull()).astype(int)
    person_df[person_df['label']==1]

    # In[ ]:
    """
        (6) Rename the column name of the loaded table
    """
    meas_df.rename(columns={'measurement_concept_id':'concept_id','measurement_date':'concept_date','value_as_number':'concept_value'}, inplace=True)
    drug_df.rename(columns={'drug_concept_id':'concept_id','drug_exposure_start_date':'concept_date','quantity':'concept_value'}, inplace=True)
    proc_df.rename(columns={'procedure_concept_id':'concept_id','procedure_date':'concept_date'}, inplace=True)
    cond_df.rename(columns={'condition_concept_id':'concept_id','condition_start_date':'concept_date'}, inplace=True)

    # In[ ]:
    """
        (7) Remove duplicate data
    """
    def drop_duplicates_(domain_df):
        n_prev = len(domain_df)
        domain_df = domain_df.drop_duplicates()
        n_next = len(domain_df)
        print('{}>{}'.format(n_prev, n_next))
        return domain_df

    meas_df = drop_duplicates_(meas_df)
    drug_df = drop_duplicates_(drug_df)
    proc_df = drop_duplicates_(proc_df)
    cond_df = drop_duplicates_(cond_df)

    # In[ ]:
    """
        (8) join Table ; Person + Domain + concept
    """
    ### domain = person + domain
    meas_df = pd.merge(person_df, meas_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
    drug_df = pd.merge(person_df, drug_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
    proc_df = pd.merge(person_df, proc_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)
    cond_df = pd.merge(person_df, cond_df, left_on=["person_id"], right_on=["person_id"], how="inner").reset_index(drop=True)

    ### domain = domain + concept
    meas_df = pd.merge(meas_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")
    drug_df = pd.merge(drug_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")
    proc_df = pd.merge(proc_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")
    cond_df = pd.merge(cond_df,concept_df,left_on=["concept_id"],right_on=["concept_id"],how="inner")

    # In[ ]:
    """
        (9) Filter by data within the observation period
    """
    meas_df = cohortConditionSetting(meas_df, pre_observation_period=60, post_observation_peroid=60, delDataAfterAbnormal=True)
    drug_df = cohortConditionSetting(drug_df, pre_observation_period=60, post_observation_peroid=60, delDataAfterAbnormal=True)
    proc_df = cohortConditionSetting(proc_df, pre_observation_period=60, post_observation_peroid=60, delDataAfterAbnormal=True)
    cond_df = cohortConditionSetting(cond_df, pre_observation_period=60, post_observation_peroid=60, delDataAfterAbnormal=True)

    # make concept_domain column before concat.
    meas_df['concept_domain'] = 'meas'
    drug_df['concept_domain'] = 'drug'
    proc_df['concept_domain'] = 'proc'
    cond_df['concept_domain'] = 'cond'
    
    # Binaryization of drug administration and diagnosis
    drug_df['concept_value'] = 1
    proc_df['concept_value'] = 1
    cond_df['concept_value'] = 1

    # @ todo : 간독성 신독성 구분. 
    all_domain_df = pd.concat([meas_df, drug_df, proc_df, cond_df], axis=0, ignore_index=True)
    
    # In[ ]:
    """
        (10) Finally Save all domain table
    """    
    all_domain_df.to_csv(output_data_dir.joinpath('all_domain_df.txt'),index=False)
    # all_domain_baseline_df = all_domain_df.query('cohort_start_date >= concept_date')
    # all_domain_baseline_df.to_csv(output_data_dir.joinpath('all_domain_baseline_df.txt'),index=False)

    # Check the number of patients
    n_label1 = len(all_domain_df[all_domain_df['label']==1].person_id.unique())
    n_label0 = len(all_domain_df[all_domain_df['label']==0].person_id.unique())
    print('label 1 : ', n_label1)
    print('label 0 : ', n_label0)

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
conn.close()

# %%
