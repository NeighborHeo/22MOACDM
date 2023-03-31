import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel, SelectPercentile, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def intersect(a, b):
    return list(set(a) & set(b))

def getPairedTTest(baseline_df, abnormal_df, concept_list):
    baseline_df = baseline_df[baseline_df['label']==1]
    abnormal_df = abnormal_df[abnormal_df['label']==1]
    import scipy.stats
    selected_var_df = pd.DataFrame()
    concept_set = list(set(baseline_df.columns) & set(abnormal_df.columns) & set(concept_list))
    # print(len(concept_set), concept_set)
    for concept in concept_set:
        # print(abnormal_df[concept].mean(), baseline_df[concept].mean())
        statistic, pvalue = scipy.stats.ttest_ind(abnormal_df[concept], baseline_df[concept], equal_var=False, nan_policy='omit')
        label_1_before = len(baseline_df[concept].dropna()) 
        label_1_after = len(abnormal_df[concept].dropna()) 
        label_1_before_mean = baseline_df[concept].dropna().mean() 
        label_1_after_mean = abnormal_df[concept].dropna().mean()
        # print(concept, pvalue)
        if statistic>1 and pvalue<0.05 :
            # print(concept)
            var_temp = {}
            var_temp['concept_id'] = concept
            var_temp['pvalue'] = pvalue
            var_temp['label_1_before'] = label_1_before
            var_temp['label_1_after'] = label_1_after
            var_temp['label_1_before_mean'] = label_1_before_mean
            var_temp['label_1_after_mean'] = label_1_after_mean
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
    return selected_var_df

def getMcnemarTest(baseline_df, abnormal_df, concept_list):
    import scipy.stats
    from statsmodels.stats.contingency_tables import mcnemar
    selected_var_df = pd.DataFrame()
    concept_set = list(set(baseline_df.columns) & set(abnormal_df.columns) & set(concept_list))
    # print(len(concept_set), concept_set)
    for concept in concept_set:
        label_0_before = len(baseline_df[(baseline_df['label']==0) & (baseline_df[concept]==1)])
        label_1_before = len(baseline_df[(baseline_df['label']==1) & (baseline_df[concept]==1)])
        label_0_after = len(abnormal_df[(abnormal_df['label']==0) & (abnormal_df[concept]==1)]) 
        label_1_after = len(abnormal_df[(abnormal_df['label']==1) & (abnormal_df[concept]==1)]) 
        arr_before = np.array([label_1_before, label_0_before])
        arr_after = np.array([label_1_after, label_0_after])
        table = np.vstack([arr_before, arr_after]) # vertical stack
        table = np.transpose(table)             # trans pose
        result = mcnemar(table, exact=True) # 샘플 수<25 일 경우 mcnemar(table, exact=False, correction=True)
        if result.pvalue < 0.05 :
            # print(concept)
            var_temp = {}
            var_temp['concept_id'] = concept
            var_temp['pvalue'] = result.pvalue
            var_temp['label_0_before'] = label_0_before
            var_temp['label_0_after'] = label_0_after
            var_temp['label_1_before'] = label_1_before
            var_temp['label_1_after'] = label_1_after
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
    return selected_var_df

def average_duration_of_adverse_events(df):
    df = df[['person_id', 'cohort_start_date', 'first_abnormal_date']].drop_duplicates() #.subject_id.unique()
    df['c_f'] = df['first_abnormal_date'] - df['cohort_start_date']
    # print(df['c_f'].describe())
    return df['c_f'].mean().days

def make_pivot(df):
    if df.empty:
        return pd.DataFrame()
    print("person_id(count) : ", df.person_id.nunique(), "concept_name(count) : ", df.concept_name.nunique())
    df = df.sort_values(by=['person_id', 'concept_id', 'concept_date'], axis=0, ascending=True)
    df['first_abnormal_date'] = pd.to_datetime(df['first_abnormal_date']).fillna(pd.to_datetime('1900-01-01'))
    last_record_df = df.groupby(by=['person_id', 'concept_id']).apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    def subtract(x, y):
        return [item for item in x if item not in set(y)]
    pivot_cols = subtract(last_record_df.columns, ['concept_name', 'concept_date', 'concept_id', 'concept_value', 'concept_domain'])
    pivot_df = pd.pivot_table(data = last_record_df, index = pivot_cols, columns='concept_id', values='concept_value').reset_index()
    return pivot_df

def impute_conditional_data(df, concept_ids):
    cols = list(set(df.columns)&set(concept_ids))
    df[cols] = df[cols].fillna(df[cols].median())
    return df
    
def impute_binary_data(df, concept_ids):
    cols = list(set(df.columns)&set(concept_ids))
    df[cols] = df[cols].fillna(0)
    return df

def normalization_Robust(df):
    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler()
    transformer.fit(df)
    df = transformer.transform(df) 
    return df 

def normalization_std(df):
    from sklearn.preprocessing import StandardScaler
    transformer = StandardScaler()
    transformer.fit(df)
    df = transformer.transform(df) 
    return df 

def normalization_minmax(df):
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler()
    transformer.fit(df)
    df = transformer.transform(df) 
    return df 

def add_column_concept_name(df, concept_id_name_dict):
    df['concept_name'] = df.apply(lambda x: concept_id_name_dict[x.concept_id] if x.concept_id in concept_id_name_dict.keys() else x.concept_id, axis = 1)
    return df

def add_column_concept_domain(df, concept_id_domain_dict):
    df['concept_domain'] = df.apply(lambda x: concept_id_domain_dict[x.concept_id] if x.concept_id in concept_id_domain_dict.keys() else 'common', axis = 1)
    print(df.concept_domain.value_counts())
    return df

def make_concepts_df(selected_features, concept_id_name_dict, concept_id_domain_dict):
    if len(selected_features) < 1:
        return pd.DataFrame()
        
    df = pd.DataFrame(selected_features, columns =['concept_id'])
    df = add_column_concept_name(df, concept_id_name_dict)
    df = add_column_concept_domain(df, concept_id_domain_dict)
    print(len(df.concept_id.unique()))
    return df

def write_file_method(df, dir, name, method):
    if df.empty:
        return False
    full_file_path = pathlib.Path('{}/{}_{}.csv'.format(dir, name, method))
    df.to_csv(full_file_path, index=False, float_format='%g')
    return True

def read_files_method(dir, name, method):
    full_file_path = pathlib.Path('{}/{}_{}.csv'.format(dir, name, method))
    if not pathlib.Path.exists(full_file_path):
        return pd.DataFrame()
    df = pd.read_csv(full_file_path)
    return df

def read_files_all_methods(dir, name):
    methods = ['statistics', 'VT', 'KBest', 'percentile', 'ExtraTrees', 'lasso_0_1', 'lasso_0_0_1', 'mutual']
    concat_df = pd.DataFrame()
    for method in methods:
        method_df = read_files_method(dir, name, method)
        if method_df.empty:
            continue
        method_df['method'] = method
        concat_df = pd.concat([concat_df, method_df], axis=0)
    return concat_df

def resumetable(df):
    print(f'data frame shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['data_type'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'feature'})
    summary['n_values'] = df.notnull().sum().values
    summary['n_missingvalues'] = df.isnull().sum().values
    summary['n_missingrate'] = df.isnull().sum().values/len(df)
    summary['n_eigenvalues'] = df.nunique().values
    return summary

def getxydata(df):
    x_df = df.drop(['person_id', 'cohort_start_date', 'first_abnormal_date', 'label'], axis=1) 
    # x_df = (x_df-x_df.min())/(x_df.max()-x_df.min()) # normalize
    x_data = x_df.to_numpy()
    y_data = df['label'].to_numpy()
    cols = x_df.columns
    return x_data, y_data, cols

def concat_all_methods(feature_selection_method_df_dict):
    concat_df = pd.DataFrame()
    for method in feature_selection_method_df_dict.keys():
        method_df = feature_selection_method_df_dict[method]
        if method_df.empty:
            continue
        method_df['method'] = method
        concat_df = pd.concat([concat_df, method_df], axis=0)
    return concat_df

def merge_summary_table_df(summary_df, concat_df):
    pivot_df = concat_df[['concept_id', 'concept_name', 'concept_domain', 'method']]
    pivot_df['value'] = 1
    pivot_df = pd.pivot_table(data=pivot_df, columns='method', index=['concept_id', 'concept_name', 'concept_domain'], values='value', fill_value=0).reset_index()
    pivot_df['total'] = pivot_df[list(set(concat_df.method.unique()) & set(pivot_df.columns))].sum(axis=1)
    pivot_df
    pivot_df.concept_id = pivot_df.concept_id.apply(lambda _: str(_).replace('.0',''))
    summary_df.concept_id = summary_df.concept_id.apply(lambda _: str(_).replace('.0',''))
    pivot_join_df = pd.merge(left=pivot_df, right=summary_df, left_on=['concept_id'], right_on=['concept_id'], how='left')
    # old_columns = pivot_join_df.columns.to_list()
    # new_columns = ['{}_{}'.format(col,hospital[0]) for col in pivot_join_df.columns]
    # pivot_join_df.rename(dict(zip(old_columns, new_columns)), axis=1, inplace=True)
    return pivot_join_df