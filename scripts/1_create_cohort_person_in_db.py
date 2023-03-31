# %%
"""
## 1. Create Cohort (Total / Case) in your DBMS
* Prerequisites
  - In the **../config.json** file, you need to modify the schema and DB information according to your DBMS.

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
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from tqdm import tqdm
from datetime import timedelta

from _utils.customlogger import customlogger as CL

pd.set_option('display.max_colwidth', -1)    #각 컬럼 width 최대로 
pd.set_option('display.max_rows', 50)        # display 50개 까지 

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
3) set path & make directory
"""
working_date = cfg["working_date"]
curr_file_name = os.path.splitext(os.path.basename(os.path.abspath('')))[0]
output_dir = pathlib.Path('{}/result/{}/create_cohort/'.format(parent_dir, working_date))
pathlib.Path.mkdir(output_dir, mode=0o777, parents=True, exist_ok=True)

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
    conn.autocommit = True
    log.debug("postgresql connect")
    
elif (cfg["dbms"]=="mssql"):
    db_cfg = cfg["mssql"]
    import pymssql
    conn= pymssql.connect(server=db_cfg['@server'], user=db_cfg['@user'], password=db_cfg['@password'], port=db_cfg['@port'], database=db_cfg['@database'], as_dict=False)
    log.debug("mssql connect")
    
else:
    log.warning("set config.json - sql - dbms : mssql or postgresql")

# %%
# In[ ]:
"""
6) define functions 
"""
def writefile(filepath, text):
    abs_path = os.path.abspath(os.path.dirname(filepath))
    pathlib.Path.mkdir(pathlib.Path(abs_path), mode=0o777, parents=True, exist_ok=True)
    f = open(filepath, 'w')
    f.write(text)
    f.close()
    
def readfile(filepath):
    f = open(filepath, 'r')
    text = f.read()
    f.close()
    return text

def executeQuery(conn, sql_query):
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
        conn.commit()
    except:
        traceback.print_exc()
        log.error(traceback.format_exc())

def executeQuerynfetchone(conn, sql_query):
    result = None
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchone()
            if result != None:
                result = result[0]
        conn.commit()
    except:
        traceback.print_exc()
        log.error(traceback.format_exc())
    return result

def renderTranslateQuerySql(sql_query, dict):
    """
    ----- Replace text ------
    "@cdm_database_schema",
    "@target_database_schema",
    "@target_cohort_table",
    "@vocabulary_database_schema",
    "@target_cohort_id"
    ----- Replace text ------
    """
    for key, value in dict.items():
        sql_query = sql_query.replace(key, value)
    return sql_query

def checkemptyvalueindict(dict):
    for key in dict:
        if not dict[key]:
            return True
    return False

# %%
# In[ ]:
"""
7) check drug concepts id 
   : Concept ids for drugs used in the DB may be different for each institution. 
     let's first look up the concept id by drug name.
"""
sql_file_path = '../_sql/5_search_concept_ids/search_concept_ids_{dbms}.sql'
sql_file_path = sql_file_path.replace("{dbms}", cfg["dbms"])

dict_list = []
output = pd.DataFrame()
for drug in cfg['drug'].keys():
    param_dict={}
    param_dict['@cohort_database_schema'] = db_cfg['@cdm_database_schema']
    param_dict['@drugname'] = drug
    sql_query = readfile(sql_file_path)
    sql_query = renderTranslateQuerySql(sql_query, param_dict)
    # print(sql_query)
    print(param_dict)
    writefile(filepath=sql_file_path.replace('../_sql/','../query/{}/'.format(drug)), text=sql_query)
    result = executeQuerynfetchone(conn, sql_query)
    dict_list.append({'drug': drug, 'concept_id': result})
    print(result)

pd.DataFrame(dict_list).to_csv("{}/check_concept_id.txt".format(output_dir), index=False)

# %%
# In[ ]:
"""
8) create temp_cohort table 
   : Before generating cohorts by query,
     Create a table to store the created cohort id.
     for example. TEMP_MOA.Temp_Cohort
"""
for file_path in cfg['translatequerysql0']:
    param_dict = cfg['translatequerysql0'][file_path]
    sql_file_path = file_path.replace("{dbms}", cfg["dbms"])
    print(sql_file_path)
    print(param_dict)
    sql_query = readfile(sql_file_path)
    sql_query = renderTranslateQuerySql(sql_query, param_dict)
    result = executeQuery(conn, sql_query)
    writefile(filepath=sql_file_path.replace('../_sql/','../query/'), text=sql_query)

# %%
# In[ ]:
"""
9) create a cohort table of patients on medication.
   : Get and execute the query extracted by designing the cohort from ATLAS.
     The cohort patient(total) table is created in the specified schema while changing the IDs for each drug.
"""
for file_path in cfg['translatequerysql1']:
    param_dict = cfg['translatequerysql1'][file_path]
    for drug in cfg['drug'].keys():
        sql_param_dict = param_dict.copy()
        for param_key, param_value in sql_param_dict.items():
            temp_param_value = param_value
            temp_param_value = temp_param_value.replace("{drug}", drug)
            temp_param_value = temp_param_value.replace("{drug_target_cohort_id}", cfg['drug'][drug]["drug_target_cohort_id"])
            temp_param_value = temp_param_value.replace("{@drug_concept_set}", cfg['drug'][drug]["@drug_concept_set"])
            sql_param_dict[param_key] = temp_param_value

        if checkemptyvalueindict(sql_param_dict):
            continue
        
        sql_file_path = file_path.replace("{ade}", cfg['drug'][drug]["ade"])
        sql_file_path = sql_file_path.replace("{dbms}", cfg["dbms"])
        sql_file_path = sql_file_path.replace("{drug}", drug)
        print(sql_file_path)
        print(sql_param_dict)
        sql_query = readfile(sql_file_path)
        sql_query = renderTranslateQuerySql(sql_query, sql_param_dict)
        result = executeQuery(conn, sql_query)
        writefile(filepath=sql_file_path.replace('../_sql/','../query/{}/'.format(drug)), text=sql_query)

# %%
# In[ ]:
"""
10) Create a cohort table of adverse events(nephrotoxicity/hepatotoxicity) patients.
   : Get and execute the Outcome (nephrotoxicity/hepatotoxicity) query from ATLAS to design and extract cohorts.
     Creates a outcome table in the specified schema.
"""
for file_path in cfg['translatequerysql2']:
    param_dict = cfg['translatequerysql2'][file_path]
    for i, ade in enumerate(['nephrotoxicity', 'hepatotoxicity']):
        sql_param_dict = param_dict.copy()
        for param_key, param_value in sql_param_dict.items():
            temp_param_value = param_value
            temp_param_value = temp_param_value.replace("{drug_target_cohort_id}", '{}'.format(99990+i))
            temp_param_value = temp_param_value.replace("{ade}", ade)
            sql_param_dict[param_key] = temp_param_value
        if checkemptyvalueindict(sql_param_dict):
            continue
        
        sql_file_path = file_path.replace("{ade}", ade)
        sql_file_path = sql_file_path.replace("{dbms}", cfg["dbms"])
        print(sql_file_path)
        print(sql_param_dict)
        sql_query = readfile(sql_file_path)
        sql_query = renderTranslateQuerySql(sql_query, sql_param_dict)
        result = executeQuery(conn, sql_query)
        writefile(filepath=sql_file_path.replace('../_sql/','../query/'), text=sql_query)

# %%
# In[]:
"""
12) Create a person table with the outcome of the drug taking subject.
    : Combine the Outcome Table and Drug Cohort Table created above.
"""
for file_path in cfg['translatequerysql3']:
    param_dict = cfg['translatequerysql3'][file_path]
    for drug in cfg['drug'].keys():
        sql_param_dict = param_dict.copy()
        for param_key, param_value in sql_param_dict.items():
            temp_param_value = param_value
            temp_param_value = temp_param_value.replace("{drug}", drug)
            temp_param_value = temp_param_value.replace("{ade}", cfg['drug'][drug]["ade"])
            sql_param_dict[param_key] = temp_param_value
        sql_file_path = file_path.replace("{dbms}", cfg["dbms"]).replace("{drug}", drug)
        print(sql_file_path)
        print(sql_param_dict)
        sql_query = readfile(sql_file_path)
        sql_query = renderTranslateQuerySql(sql_query, sql_param_dict)
        result = executeQuery(conn, sql_query)
        writefile(filepath=sql_file_path.replace('../_sql/','../query/{}/'.format(drug)), text=sql_query)

# %%
# In[ ]:
"""
13) Check the number of patients
    : total N of patients (taking the drug) /  N of patients (with side effects)
"""
dict_list = []
for drug in cfg['drug'].keys():
    sql_query = "select count(distinct person_id) from {}.person_{} where n_diff is null".format(db_cfg['@person_database_schema'], drug)
    # print("select * from person_{}".format(drug))
    n_total_population = executeQuerynfetchone(conn, sql_query)
    
    sql_query = "select count(distinct person_id) from {}.person_{} where n_diff is not null".format(db_cfg['@person_database_schema'], drug)
    # print("select * from person_{}".format(drug))
    n_case_population = executeQuerynfetchone(conn, sql_query)
    
    dict_list.append({'drug': drug, 'n_total_population': n_total_population, 'n_case_population': n_case_population})
    
nPatients_df = pd.DataFrame(dict_list)
nPatients_df.to_csv("{}/check_numberofpatient.txt".format(output_dir), index=False)
print(nPatients_df)

# %%
# In[ ]:
conn.close()

# %%
