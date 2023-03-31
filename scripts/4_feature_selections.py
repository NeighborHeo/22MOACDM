# %%
"""
## 4. Selection of Predictor variable
  * Select predictor variables using various variable selection methods

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
from _utils.feature_selection import *
from _utils.customlogger import customlogger as CL

%matplotlib inline

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
# In[ ]:
def runTask(outcome_name):
    # In[ ]:
    """
        (1) set path & make directory
    """
    importsql_data_dir   = pathlib.Path('{}/data/{}/importsql/{}/'.format(parent_dir, current_date, outcome_name))
    output_data_dir      = pathlib.Path('{}/data/{}/feature_selection/{}/'.format(parent_dir, current_date, outcome_name))
    output_result_dir    = pathlib.Path('{}/result/{}/feature_selection/{}/'.format(parent_dir, current_date, outcome_name))
    pathlib.Path.mkdir(output_data_dir, mode=0o777, parents=True, exist_ok=True)
    pathlib.Path.mkdir(output_result_dir, mode=0o777, parents=True, exist_ok=True)

    # In[ ]:
    """
        (2) read data
    """
    if not pathlib.Path.exists(importsql_data_dir.joinpath('all_domain_df.txt')):
        log.debug(f"{outcome_name} data frame is empty")
        return
    
    all_domain_df = pd.read_csv(importsql_data_dir.joinpath('all_domain_df.txt'), low_memory=False)
    nCase = all_domain_df.loc[all_domain_df['label'] == 1].person_id.nunique()
    if nCase < 20:
        log.debug(f"{outcome_name} case is less than 20")
        return

    # In[ ]:
    """
        (3) use only necessary columns
    """
    common_cols = ['person_id', 'age', 'sex', 'cohort_start_date', 'first_abnormal_date', 'concept_date', 'concept_id', 'concept_name', 'concept_value', 'concept_domain', 'label']
    all_domain_df = all_domain_df[common_cols]
    
    # In[ ]:
    """
        (4) Remove feature used in outcome define
    """
    drug_name = outcome_name
    drug_concept_ids_excluded = map(int,cfg['drug'][drug_name]['@drug_concept_set'].split(','))
    all_domain_df = all_domain_df.loc[~all_domain_df.concept_id.isin(drug_concept_ids_excluded)]
    meas_concept_ids_excluded = map(int,[cfg['meas'][meas_name]['@meas_concept_id'] for meas_name in cfg['meas']])
    all_domain_df = all_domain_df.loc[~all_domain_df.concept_id.isin(meas_concept_ids_excluded)]
    
    all_domain_df['cohort_start_date'] = pd.to_datetime(all_domain_df['cohort_start_date'], infer_datetime_format=True)
    all_domain_df['first_abnormal_date'] = pd.to_datetime(all_domain_df['first_abnormal_date'], infer_datetime_format=True)
    all_domain_df['concept_date'] = pd.to_datetime(all_domain_df['concept_date'], infer_datetime_format=True)

    # In[ ]:
    """
        (5) Check the average duration of occurrence until side effects
    """
    ndays = average_duration_of_adverse_events(all_domain_df)
    log.debug('average_duration_of_adverse_events : {}'.format(ndays))

    # In[ ]:
    """
        (6) make baseline data
    """
    all_domain_pivot_df = make_pivot(all_domain_df)
    all_domain_baseline_df = all_domain_df.query('cohort_start_date >= concept_date')
    all_domain_pivot_baseline_df = make_pivot(all_domain_baseline_df)

    summary_df = resumetable(all_domain_pivot_df)
    write_file_method(summary_df, output_result_dir, outcome_name, 'summary')

    concept_id_name_dict = dict(zip(all_domain_df.concept_id, all_domain_df.concept_name))
    concept_id_domain_dict = dict(zip(all_domain_df.concept_id, all_domain_df.concept_domain))

    feature_selection_method_df_dict = {}
    
    
    # In[ ]:
    """
        (7) Feature_Selection Method 1 : statistics method
    """
    meas_concept_ids = list(all_domain_df.loc[all_domain_df.concept_domain=='meas'].concept_id.values)
    drug_concept_ids = list(all_domain_df.loc[all_domain_df.concept_domain=='drug'].concept_id.values)
    proc_concept_ids = list(all_domain_df.loc[all_domain_df.concept_domain=='proc'].concept_id.values)
    cond_concept_ids = list(all_domain_df.loc[all_domain_df.concept_domain=='cond'].concept_id.values)

    selected_features_with_t_test_df = getPairedTTest(all_domain_pivot_baseline_df, all_domain_pivot_df, meas_concept_ids)
    selected_features_with_mcnemar_df = getMcnemarTest(all_domain_pivot_baseline_df, all_domain_pivot_df, drug_concept_ids + cond_concept_ids + proc_concept_ids)

    selected_features_df = pd.concat([selected_features_with_t_test_df, selected_features_with_mcnemar_df], axis=0)
    if not selected_features_df.empty:
        selected_features_df.concept_id = selected_features_df.concept_id.astype(np.object)
    selected_features_df = add_column_concept_name(selected_features_df, concept_id_name_dict)
    selected_features_df = add_column_concept_domain(selected_features_df, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'statistics')
    feature_selection_method_df_dict['statistics'] = selected_features_df

    len(all_domain_df.concept_id.unique()), len(all_domain_baseline_df.concept_id.unique())

    # In[ ]:
    """
        (8) imputation missing data
    """
    all_domain_pivot_df = impute_conditional_data(all_domain_pivot_df, meas_concept_ids)
    all_domain_pivot_df = impute_binary_data(all_domain_pivot_df, drug_concept_ids + proc_concept_ids + cond_concept_ids)

    meas_concept_ids = list(all_domain_baseline_df.loc[all_domain_baseline_df.concept_domain=='meas'].concept_id.values)
    drug_concept_ids = list(all_domain_baseline_df.loc[all_domain_baseline_df.concept_domain=='drug'].concept_id.values)
    proc_concept_ids = list(all_domain_baseline_df.loc[all_domain_baseline_df.concept_domain=='proc'].concept_id.values)
    cond_concept_ids = list(all_domain_baseline_df.loc[all_domain_baseline_df.concept_domain=='cond'].concept_id.values)

    all_domain_pivot_baseline_df = impute_conditional_data(all_domain_pivot_baseline_df, meas_concept_ids)
    all_domain_pivot_baseline_df = impute_binary_data(all_domain_pivot_baseline_df, drug_concept_ids + proc_concept_ids + cond_concept_ids)

    X_total, y_total, cols = getxydata(all_domain_pivot_df)
    X_total = normalization_minmax(X_total)

    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=1, stratify=y_total) 

    # check for nan / infinite value 
    np.argwhere(np.isnan(X_train)), np.argwhere(np.isinf(X_train))

    # In[ ]:
    """
        (9) Feature_Selection Method 2 : VarianceThreshold
    """
    selector = VarianceThreshold(1e-3)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    print(X_train.shape, X_train_sel.shape)
    selected_features = selector.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'VT')
    feature_selection_method_df_dict['VT'] = selected_features_df

    # In[ ]:
    """
        (10) Feature_Selection Method 3 : SelectPercentile
    """
    selector = SelectPercentile(chi2, percentile=3) # now select features based on top 10 percentile
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(X_train.shape, X_train_sel.shape)
    selected_features = selector.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'percentile')
    feature_selection_method_df_dict['percentile'] = selected_features_df

    # In[ ]:
    """
        (11) Feature_Selection Method 4 : SelectPercentile
    """
    selector = SelectKBest(score_func=chi2, k=50)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(X_train.shape, X_train_sel.shape)
    selected_features = selector.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'KBest')
    feature_selection_method_df_dict['KBest'] = selected_features_df

    # In[ ]:
    """
        (12) Feature_Selection Method 5 : ExtraTreesClassifier
    """
    treebasedclf = ExtraTreesClassifier(n_estimators=50)
    treebasedclf = treebasedclf.fit(X_train, y_train)
    model = SelectFromModel(treebasedclf, prefit=True)
    X_train_sel = model.transform(X_train)
    print(X_train.shape, X_train_sel.shape)
    selected_features = model.get_feature_names_out(cols)
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'ExtraTrees')
    feature_selection_method_df_dict['ExtraTrees'] = selected_features_df

    # In[ ]:
    """
        (13) Feature_Selection Method 6 : Lasso (1) > alpha = 0.1
    """
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    # print(lasso.coef_)
    importance = np.abs(lasso.coef_)
    selected_features = np.array(cols)[importance > 0]
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)

    write_file_method(selected_features_df, output_result_dir, outcome_name, 'lasso_0_1')
    feature_selection_method_df_dict['lasso_0_1'] = selected_features_df

    # In[ ]:
    """
        (14) Feature_Selection Method 7 : mutual_info_classif
    """
    importances = mutual_info_classif(X_total, y_total, discrete_features='auto')
    threshold = 0.001
    selected_features = np.array(cols)[importance > threshold]
    selected_features_df = make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict)
    write_file_method(selected_features_df, output_result_dir, outcome_name, 'mutual')
    feature_selection_method_df_dict['mutual'] = selected_features_df

    # In[ ]:
    """
        (15) all methods concatenate
    """
    concat_df = concat_all_methods(feature_selection_method_df_dict)
    write_file_method(concat_df, output_result_dir, outcome_name, 'all_methods')
    summary_concat_df = merge_summary_table_df(summary_df, concat_df)
    write_file_method(summary_concat_df, output_result_dir, outcome_name, 'total')
    


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
