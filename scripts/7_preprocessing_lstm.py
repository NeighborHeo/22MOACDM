# %%
"""
## 4. Preprocessing for LSTM
  * Preprocessing with TimeSeries data format For use as input to the LSTM model

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
from _utils.preprocessing_lstm import *
from _utils.customlogger import customlogger as CL

# %%
# In[ ]:
"""
    2) loading config
"""
current_dir = pathlib.Path.cwd()
parent_dir = current_dir.parent

with open(parent_dir.joinpath("config.json")) as file:
    cfg = json.load(file)
with open(parent_dir.joinpath("config_params.json")) as file:
    params = json.load(file)

# %%
# In[ ]:
"""
    3) load information 
"""
current_date = cfg["working_date"]
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
def resampling_4w(df):
    return df.groupby('unique_id').apply(lambda x : x.set_index('concept_date').resample('7d').last().reset_index()).reset_index(drop=True)

def get_time_stamp(df):
    return int(len(df)/len(df.unique_id.unique()))

def remove_special_characters(str):
    import re
    return re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', str)

def save_single_concept_bar_plot(dir, df, concept_id, concept_name):
    import seaborn as sns
    import textwrap
    fig=plt.figure()
    plt.rcParams['figure.figsize'] = [7.00, 3.50]
    plt.rcParams['figure.autolayout'] = True
    ax = sns.boxplot(data=df, x="sequence", y=concept_id, hue='label')
        # showcaps=False,                     # 박스 상단 가로라인 보이지 않기
        # whiskerprops={'linewidth':0},       # 박스 상단 세로 라인 보이지 않기 
        # showfliers=True                     # 박스 범위 벗어난 아웃라이어 표시하지 않기
    concept_name_short = textwrap.shorten(concept_name, width=60, placeholder="...")
    ax.set(title='{} ( {} )'.format(concept_name_short, concept_id))
    timestamp = get_time_stamp(df)
    ax.set_xticklabels(['t - {}w'.format(i) for i in range(timestamp, 0, -1)])
    plt.xlabel('time')
    plt.ylabel('value')
    
    plt.savefig('{}/{}.png'.format(dir, concept_name), format='png',
            dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
    plt.show()
        
def save_all_features(dir, df, concept_dict):
    concept_id_list = list(set(df.columns) & set(concept_dict.keys()))
    for concept_id in concept_id_list:
        concept_name = remove_special_characters(concept_dict[concept_id])
        save_single_concept_bar_plot(dir, df, concept_id, concept_name)

# %%
def runTask(outcome_name):
    # In[ ]:
    """
        (1) Set the folder path to load and save data
    """
    importsql_data_dir    = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, current_date, outcome_name))           # input file path
    output_data_dir         = pathlib.Path('{}/data/{}/preprocess_lstm/{}/'.format(parent_dir, current_date, outcome_name))     # output file path
    output_result_dir       = pathlib.Path('{}/result/{}/preprocess_lstm/{}/'.format(parent_dir, current_date, outcome_name))   # output file path (features)
    pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
    pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

    # In[ ]:
    """
        (2) Skip if there is no patient data
    """
    if not pathlib.Path.exists(importsql_data_dir.joinpath('all_domain_df.txt')):
        log.debug(f"{outcome_name} data frame is empty")
        return
    
    # In[ ]:
    """
        (3) read data frame
    """
    all_domain_df = pd.read_csv(importsql_data_dir.joinpath('all_domain_df.txt'), low_memory=False)
    print(all_domain_df.concept_domain.value_counts())
    print('label 1 : ', len(all_domain_df[all_domain_df['label']==1].person_id.unique()))
    print('label 0 : ', len(all_domain_df[all_domain_df['label']==0].person_id.unique()))
    nCase = all_domain_df.loc[all_domain_df['label'] == 1].person_id.nunique()
    if nCase < 20:
        log.debug(f"{outcome_name} case is less than 20")
        return
    
    # In[ ]:
    """
        (4) use only necessary columns
    """
    common_cols = ['person_id', 'age', 'sex', 'cohort_start_date', 'first_abnormal_date', 'concept_date', 'concept_id', 'concept_name', 'concept_value', 'concept_domain', 'label']
    all_domain_df = all_domain_df[common_cols]

    # In[ ]:
    """
        (5) Remove feature used in outcome define
    """
    drug_name = outcome_name
    drug_concept_ids_excluded = map(int,cfg['drug'][drug_name]['@drug_concept_set'].split(','))
    all_domain_df = all_domain_df.loc[~all_domain_df.concept_id.isin(drug_concept_ids_excluded)]
    meas_concept_ids_excluded = map(int,[cfg['meas'][meas_name]['@meas_concept_id'] for meas_name in cfg['meas']])
    all_domain_df = all_domain_df.loc[~all_domain_df.concept_id.isin(meas_concept_ids_excluded)]
    
    # ---------------------- check features ----------------------------
    concept_list = []
    nCaseInTotal = len(all_domain_df.loc[all_domain_df['label']==1,'person_id'].unique())
    nControlInTotal =len(all_domain_df.loc[all_domain_df['label']==0,'person_id'].unique())

    meas_df = all_domain_df.loc[all_domain_df.concept_domain=='meas'].reset_index(drop=True)
    drug_df = all_domain_df.loc[all_domain_df.concept_domain=='drug'].reset_index(drop=True)
    proc_df = all_domain_df.loc[all_domain_df.concept_domain=='proc'].reset_index(drop=True)
    cond_df = all_domain_df.loc[all_domain_df.concept_domain=='cond'].reset_index(drop=True)

    meas_df = meas_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
    drug_df = drug_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
    proc_df = proc_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
    cond_df = cond_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)

    meas_concept_df = meas_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
    drug_concept_df = drug_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
    proc_concept_df = proc_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)
    cond_concept_df = cond_df.groupby('concept_id').apply(lambda x : filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold=0.1)).reset_index(drop=True)

    meas_concept_df['concept_domain'] = 'meas'
    drug_concept_df['concept_domain'] = 'drug'
    proc_concept_df['concept_domain'] = 'proc'
    cond_concept_df['concept_domain'] = 'cond'
    
    all_domain_concept_df = pd.concat([meas_concept_df, drug_concept_df, proc_concept_df, cond_concept_df], axis=0, ignore_index=True)
    all_domain_concept_df.to_csv('{}/{}_feature_2.csv'.format(output_result_dir, outcome_name), header=True, index=True)
    # -------------------------------------------------------------------
    
    # @variable selection
    meas_vars_df = variant_selection_paired_t_test(meas_df)
    drug_vars_df = variant_selection_mcnemar(drug_df)
    proc_vars_df = variant_selection_mcnemar(proc_df)
    cond_vars_df = variant_selection_mcnemar(cond_df)

    # @variable selection (Top 30 based on p Value)
    #pd.options.display.precision = 3
    meas_vars_df = meas_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
    drug_vars_df = drug_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
    proc_vars_df = proc_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
    cond_vars_df = cond_vars_df.sort_values(by='pvalue', ascending=True).reset_index(drop=True).head(30)
    print(len(meas_vars_df), len(drug_vars_df), len(proc_vars_df), len(cond_vars_df))
    
    meas_vars_df['concept_domain'] = 'meas'
    drug_vars_df['concept_domain'] = 'drug'
    proc_vars_df['concept_domain'] = 'proc'
    cond_vars_df['concept_domain'] = 'cond'
    all_domain_vars_df = pd.concat([meas_vars_df, drug_vars_df, proc_vars_df, cond_vars_df], axis=0, ignore_index=True)
    # @variable selection (save)
    all_domain_vars_df.to_csv('{}/{}_feature.csv'.format(output_result_dir, outcome_name), header=True, index=True)
    # all_domain_vars_df = pd.read_csv('{}/{}_{}_feature.csv'.format(output_result_dir, outcome_name), index_col=False) #check

    # @Extract only selected concepts from data frame
    def extractSelectedConceptID(domain_df, concept_id_list):
        extract_domain_df = domain_df[domain_df['concept_id'].isin(concept_id_list)]
        print(len(concept_id_list), len(domain_df), len(extract_domain_df))
        return extract_domain_df

    meas_fs_df = extractSelectedConceptID(meas_df, meas_vars_df.concept_id.unique())
    drug_fs_df = extractSelectedConceptID(drug_df, drug_vars_df.concept_id.unique())
    proc_fs_df = extractSelectedConceptID(proc_df, proc_vars_df.concept_id.unique())
    cond_fs_df = extractSelectedConceptID(cond_df, cond_vars_df.concept_id.unique())

    all_domain_df = pd.concat([meas_fs_df, drug_fs_df, proc_fs_df, cond_fs_df], axis=0, ignore_index=True)

    pivot_data = pivotting(all_domain_df)

    drop_cols = []
    for col in pivot_data.columns:
        if (len(pivot_data[pivot_data[col].notnull()])/len(pivot_data[col]) < 0.3):
            drop_cols.append(col)
    print(drop_cols)
    
    pivot_data = pivot_data.drop(drop_cols, axis='columns')

    domain_ids={}
    domain_ids['meas'] = np.setdiff1d(meas_fs_df.concept_id.unique(), drop_cols)
    domain_ids['drug'] = np.setdiff1d(drug_fs_df.concept_id.unique(), drop_cols)
    domain_ids['proc'] = np.setdiff1d(proc_fs_df.concept_id.unique(), drop_cols)
    domain_ids['cond'] = np.setdiff1d(cond_fs_df.concept_id.unique(), drop_cols)

    # -------- time series data ---------
    interpolate_df = day_sequencing_interpolate(pivot_data, domain_ids, OBP=params["windowsize"])

    label_1 = interpolate_df[interpolate_df['label']==1]
    label_0 = interpolate_df[interpolate_df['label']==0]

    rolled_label1_d = shift_rolling_window(label_1, OBP=params["windowsize"], nShift=params["shift"], uid_index=1)
    rolled_label0_d = label_0_fitting(label_0, OBP=params["windowsize"], nShift=params["shift"], uid_index=(rolled_label1_d.unique_id.max()+1))

    # label 0 + label 1
    concat_df = pd.concat([rolled_label1_d, rolled_label0_d], sort=False)
    concat_df = concat_df.sort_values(['unique_id', 'concept_date'])
    # -------- time series data ---------

    concept_dict = dict(zip(all_domain_df.concept_id, all_domain_df.concept_name))
    concat_4w_df = resampling_4w(concat_df)
    draw_4w_df = concat_4w_df.copy()
    draw_4w_df['sequence'] = draw_4w_df.groupby('unique_id').cumcount()+1
    save_all_features(output_result_dir, draw_4w_df, concept_dict)

    # Normalization (Min/Max Scalar)
    concat_df = normalization(concat_df)
    concat_df = concat_df.dropna(axis=1)
    concat_4w_df = normalization(concat_4w_df)
    concat_4w_df = concat_4w_df.dropna(axis=1)

    # columns name : concept_id > concept_name
    concat_df = concat_df.rename(concept_dict, axis='columns')
    concat_4w_df = concat_4w_df.rename(concept_dict, axis='columns')

    # Save File
    concat_df.to_csv('{}/{}.txt'.format(output_data_dir, outcome_name), index=False, float_format='%g')
    concat_4w_df.to_csv('{}/{}_4w.txt'.format(output_data_dir, outcome_name), index=False, float_format='%g')

    output={}
    output['meas_whole_var'] = len(meas_df.concept_id.unique())
    output['drug_whole_var'] = len(drug_df.concept_id.unique())
    output['proc_whole_var'] = len(proc_df.concept_id.unique())
    output['cond_whole_var'] = len(cond_df.concept_id.unique())
    output['meas_selected_var'] = len(domain_ids['meas'])
    output['drug_selected_var'] = len(domain_ids['drug'])
    output['proc_selected_var'] = len(domain_ids['proc'])
    output['cond_selected_var'] = len(domain_ids['cond'])
    output['nPatient_label1'] = len(concat_df[concat_df["label"] == 1])
    output['nPatient_label0'] = len(concat_df[concat_df["label"] == 0])

    # print
    print(output['meas_whole_var'], output['meas_selected_var'])
    print(output['drug_whole_var'], output['drug_selected_var'])
    print(output['proc_whole_var'], output['proc_selected_var'])
    print(output['cond_whole_var'], output['cond_selected_var'])

    out = open('{}/output.txt'.format(output_result_dir),'a')

    out.write(str(outcome_name) + '///' )
    out.write(str(output['meas_whole_var']) + '///')
    out.write(str(output['meas_selected_var']) + '///')
    out.write(str(output['drug_whole_var']) + '///')
    out.write(str(output['drug_selected_var']) + '///')
    out.write(str(output['proc_whole_var']) + '///')
    out.write(str(output['proc_selected_var']) + '///')
    out.write(str(output['cond_whole_var']) + '///')
    out.write(str(output['cond_selected_var']) + '///')
    out.write(str(output['nPatient_label1']) + '///')
    out.write(str(output['nPatient_label0']) + '\n')
    out.close()


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
