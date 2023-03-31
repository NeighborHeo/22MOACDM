import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import os
import pandas as pd
import pymssql
from statsmodels.stats.contingency_tables import mcnemar
from datetime import timedelta
from sklearn import preprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from matplotlib import pyplot
from sklearn.metrics import classification_report, roc_curve, auc
import warnings
warnings.filterwarnings(action='ignore')

def cohortConditionSetting(domain_df, pre_observation_period, post_observation_peroid):
    from datetime import timedelta
    prev_len = len(domain_df)
    domain_df['cohort_start_date'] = pd.to_datetime(domain_df['cohort_start_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    domain_df['first_abnormal_date'] = pd.to_datetime(domain_df['first_abnormal_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    domain_df['concept_date'] = pd.to_datetime(domain_df['concept_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    # condition 1) Select patients with first adverse events within 2 months of cohort initiation.
    domain_df = domain_df[(domain_df['cohort_start_date']<=domain_df['concept_date']+timedelta(days=pre_observation_period))]
    # condition 2) Delete data before the cohort start date.
    domain_df = domain_df[(domain_df['concept_date']<=domain_df['cohort_start_date']+timedelta(days=post_observation_peroid))]
    # condition 3) Delete data after first_abnormal_date (Except when there is no first abnormal date.)
    domain_df = domain_df[~(domain_df['first_abnormal_date']<domain_df['concept_date'])]
    domain_df = domain_df[~(domain_df['first_abnormal_date']-domain_df['cohort_start_date']>timedelta(days=post_observation_peroid))]
    domain_df = domain_df.reset_index(drop=True)
    curr_len = len(domain_df)
    print('{} > {}'.format(prev_len, curr_len))
    return domain_df

def variant_selection_welch_t_test(domain_df):
    import scipy.stats
    concept_dict = dict(zip(domain_df.concept_id, domain_df.concept_name))
    domain_df = domain_df.query("concept_date <= cohort_start_date")
    domain_df = domain_df.sort_values(by=['person_id', 'concept_id', 'concept_date'], ascending=[True, True, False]) # last date 값만 사용 
    domain_df = domain_df.drop_duplicates(subset=['person_id', 'concept_id'], keep = 'first')
    # print(len(domain_df))
    person_df = domain_df[["person_id", "label"]].drop_duplicates()
    nTotalPatient = len(person_df)
    nNormal = len(person_df[person_df["label"] == 0])
    nAbnormal = len(person_df[person_df["label"] == 1])
    selected_var_df = pd.DataFrame(columns = ['concept_id', 'concept_name', 'pvalue', 'concept_1', 'concept_0'])
    for concept_id, group_df in domain_df.groupby(['concept_id']): 
        # print(group_df)
        n_post = len(group_df)
        normal_df = group_df[group_df['label']==0]
        abnormal_df = group_df[group_df['label']==1]
        nConcept_0 = len(normal_df)
        nConcept_1 = len(abnormal_df)
        # print(n_prev, n_post, nConcept_0, nConcept_1)
        # result = scipy.stats.chi2_contingency(np.array([[nConcept_0, nConcept_1], [ndiff_0, ndiff_1]]))
        statistic, pvalue = scipy.stats.ttest_ind(normal_df["concept_value"], abnormal_df["concept_value"], equal_var=False)    
        # pvalue = result[1]
        if pvalue < 0.05 and len(group_df) > nTotalPatient * 0.3 :
            var_temp = {}
            var_temp['concept_id'] = concept_id
            var_temp['concept_name'] = concept_dict[concept_id]
            var_temp['pvalue'] = pvalue
            var_temp['concept_1'] = nConcept_1
            var_temp['concept_0'] = nConcept_0
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
    return selected_var_df

def variant_selection_chisquare(domain_df):
    import scipy.stats
    domain_df = domain_df.query("concept_date <= cohort_start_date")
    concept_dict = dict(zip(domain_df.concept_id, domain_df.concept_name))
    domain_df = domain_df[["concept_id", "person_id", "label"]].drop_duplicates()
    # print(len(domain_df))
    person_df = domain_df[["person_id", "label"]].drop_duplicates()
    # print(person_df.label.value_counts())
    nTotalPatient = len(person_df)
    nNormal = len(person_df[person_df["label"] == 0])
    nAbnormal = len(person_df[person_df["label"] == 1])
    selected_var_df = pd.DataFrame(columns = ['concept_id', 'concept_name', 'pvalue', 'concept_1', 'concept_0'])
    for concept_id, group_df in domain_df.groupby(['concept_id']): 
        # print(group_df)
        n_prev = len(group_df)
        group_df = group_df[["person_id", "label"]].drop_duplicates()
        n_post = len(group_df)
        nConcept_0 = len(group_df[group_df['label']==0])
        nConcept_1 = len(group_df[group_df['label']==1])
        ndiff_0 = nNormal - nConcept_0
        ndiff_1 = nAbnormal - nConcept_1
        # print(n_prev, n_post, nConcept_0, nConcept_1)
        result = scipy.stats.chi2_contingency(np.array([[nConcept_0, nConcept_1], [ndiff_0, ndiff_1]]))
        pvalue = result[1]
        if pvalue < 0.05 and len(group_df) > nTotalPatient * 0.3 :
            var_temp = {}
            var_temp['concept_id'] = concept_id
            var_temp['concept_name'] = concept_dict[concept_id]
            var_temp['pvalue'] = pvalue
            var_temp['concept_1'] = nConcept_1
            var_temp['concept_0'] = nConcept_0
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
    return selected_var_df

# 3) feature selection by domain 
import numpy as np
from datetime import timedelta
from statsmodels.stats.contingency_tables import mcnemar

def contigency_generate(var, dataset):
    for concept_name, group_data in dataset.groupby(['concept_name']):               
        if concept_name == var:
            label_0 = len(group_data[group_data['label']==0 ]) 
            label_1 = len(group_data[group_data['label']==1 ]) 
            break
    return concept_name, label_0, label_1
    
def variable_select_dpc(intersectionVariables, concept_dict, first_IO, first_abnormal):
    selected_var_df = pd.DataFrame(columns = ['concept_id', 'concept_name', 'pvalue', 'fid_label1', 'fid_label0', 'ab_label1', 'ab_label0', 'nPatientST', 'nPatientAB'])
    for var in intersectionVariables:
        concept_name1, label_0, label_1 = contigency_generate(var, first_IO)
        IO_ar = np.array([label_1, label_0])

        concept_name2, label_0, label_1 = contigency_generate(var, first_abnormal)
        abnormal_ar = np.array([label_1, label_0])

        table = np.vstack([IO_ar, abnormal_ar]) # vertical stack
        table = np.transpose(table)             # trans pose

        # calculate mcnemar test
        result = mcnemar(table, exact=True) # 샘플 수<25 일 경우 mcnemar(table, exact=False, correction=True)

        # summarize the finding    
        #print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

        nPatientST = len(first_IO[first_IO["concept_name"]==var].person_id.unique())
        nPatientAB = len(first_abnormal[first_abnormal["concept_name"]==var].person_id.unique())
        
        # interpret the p-value
        alpha = 0.05
        if result.pvalue < alpha : #or (nPatientAB + nPatientST < 50):
            var_temp = {}
            var_temp['concept_id'] = concept_dict[var]
            var_temp['concept_name'] = var
            var_temp['pvalue'] = result.pvalue
            var_temp['fid_label1'] = IO_ar[0] 
            var_temp['fid_label0'] = IO_ar[1]
            var_temp['ab_label1'] = abnormal_ar[0]
            var_temp['ab_label0'] = abnormal_ar[1]
            var_temp['nPatientST'] = nPatientST
            var_temp['nPatientAB'] = nPatientAB
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)

    #print(  len( selected_var )  )
    #b = pd.DataFrame(selected_var, columns=['drug_name','pvalue'])
    return selected_var_df

def variant_selection_mcnemar(domain_df):
    concept_dict = dict(zip(domain_df.concept_name, domain_df.concept_id))
    # 최초 IO 투여날
    first_IO = domain_df[domain_df['cohort_start_date']==domain_df['concept_date']].copy()

    # 최초 abnormal 
    first_abnormal = domain_df[domain_df['first_abnormal_date']<=domain_df['concept_date']].copy()

    # 공통변수 리스트업
    IO_nm = first_IO.concept_name.unique().tolist()
    abnormal_nm = first_abnormal.concept_name.unique().tolist()
    intersectionVariables = list(set(IO_nm) & set(abnormal_nm) )

    # 공통변수로 리세팅
    first_IO = first_IO[first_IO['concept_name'].isin(intersectionVariables)].sort_values(['concept_name'])
    first_abnormal = first_abnormal[first_abnormal['concept_name'].isin(intersectionVariables)].sort_values(['concept_name'])
    vars_ = variable_select_dpc(intersectionVariables, concept_dict, first_IO, first_abnormal)
    return vars_

# 3) feature selection by domain 
def variant_selection_m(IO_start, first_abnormal_start, intersectionVariables, concept_dict):
    IO_start_inter=IO_start[IO_start["concept_name"].isin(intersectionVariables)]
    IO_start_inter=IO_start_inter[["concept_value","concept_name"]].sort_values("concept_name").reset_index(drop=True)
    first_abnormal_start_inter = first_abnormal_start[first_abnormal_start["concept_name"].isin(intersectionVariables)]
    first_abnormal_start_inter=first_abnormal_start_inter[["concept_value","concept_name"]].sort_values("concept_name").reset_index(drop=True)
    selected_var_df = pd.DataFrame(columns=['concept_id', 'concept_name', 'statistic', 'pvalue', 'label1', 'label0', 'nPatientST', 'nPatientAB'])
    import scipy.stats # for using t-test
    for idx,var in enumerate(intersectionVariables):
        IO = IO_start_inter["concept_value"][IO_start_inter["concept_name"]==var]    
        first= first_abnormal_start_inter["concept_value"][first_abnormal_start_inter["concept_name"]==var]   
        nPatientST = len(IO_start[IO_start["concept_name"]==var].person_id.unique())
        nPatientAB = len(first_abnormal_start[first_abnormal_start["concept_name"]==var].person_id.unique())
        statistic, pvalue = scipy.stats.ttest_ind(IO, first, equal_var=False)    
        if statistic>1 and pvalue<0.05 :# and (nPatientST+nPatientAB > 50):                    
            var_temp = {}
            var_temp['concept_id'] = concept_dict[var]
            var_temp['concept_name'] = var
            var_temp['statistic'] = statistic
            var_temp['pvalue'] = pvalue
            var_temp['label1'] = len(IO)
            var_temp['label0'] = len(first)
            var_temp['nPatientST'] = nPatientST
            var_temp['nPatientAB'] = nPatientAB
            selected_var_df = selected_var_df.append(var_temp, ignore_index=True)
            
    return selected_var_df 

def variant_selection_paired_t_test(domain_df):
    concept_dict = dict(zip(domain_df.concept_name, domain_df.concept_id))
    
    #domain_df = domain_df.copy()
    domain_df['first_abnormal_date'] = pd.to_datetime(domain_df['first_abnormal_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    domain_df["concept_date"] = pd.to_datetime(domain_df["concept_date"], format='%Y-%m-%d %H:%M:%S', errors='raise')
    total_idx = domain_df.index
    # measurement_data + 7 < first_abnormal_date
    IO_start=domain_df[domain_df["concept_date"]<domain_df["first_abnormal_date"]]
    IO_start=IO_start[IO_start["first_abnormal_date"]-IO_start["concept_date"]>=timedelta(days=7)] # 4ws
    IO_start_idx=IO_start.index
    
    # # 전체에서 IO_Start 빼기
    first_abnormal_start_idx=total_idx.difference(IO_start_idx)
    first_abnormal_start=domain_df.iloc[first_abnormal_start_idx]
    first_abnormal_start=first_abnormal_start[first_abnormal_start["first_abnormal_date"].notnull()]
    first_abnormal_start=first_abnormal_start[~first_abnormal_start["first_abnormal_date"].isin(['1970-01-01']) ]
    
    IO_start_idx = IO_start.concept_name.unique().tolist()
    first_abnormal_start_idx = first_abnormal_start.concept_name.unique().tolist()
    intersectionVariables = list(set(IO_start_idx) & set(first_abnormal_start_idx))
    vars_ = variant_selection_m(IO_start, first_abnormal_start, intersectionVariables, concept_dict) ############# return variable
    #print("measurement selected variable count:", len(vars_ )  )            
    #print(pd.DataFrame(vars_, columns=['lab test','t-test','p_value']).sort_values(by=['p_value']) )
    return vars_

def average_duration_of_adverse_events(df):
    df = df[['person_id', 'cohort_start_date', 'first_abnormal_date']].drop_duplicates() #.person_id.unique()
    df['c_f'] = df['first_abnormal_date'] - df['cohort_start_date']
    print(df['c_f'].describe())
    return df['c_f'].mean().days


def pivotting(all_domain_df):    
    all_domain_df['cohort_start_date'] = pd.to_datetime(all_domain_df['cohort_start_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    all_domain_df['first_abnormal_date'] = pd.to_datetime(all_domain_df['first_abnormal_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    all_domain_df['concept_date'] = pd.to_datetime(all_domain_df['concept_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    all_domain_df = all_domain_df.fillna('1970-01-01') # 
    common_columns = ['person_id','age','sex','cohort_start_date','concept_date','first_abnormal_date','label']
    # pivotting 
    pivot_data = pd.pivot_table(all_domain_df, index = common_columns, columns = ['concept_id'], values = 'concept_value')
    pivot_data = pivot_data.sort_values(by=["person_id","concept_date"],axis=0).reset_index()
    #pivot_data = pivot_data.fillna(0) ####### 뒤에 days_diff 때문에 fill_na(0) 이 안됨
    pivot_data = pivot_data.fillna(pivot_data.groupby(['person_id']).ffill())
    pivot_data = pivot_data.drop_duplicates()
    pivot_data = pivot_data.rename_axis(None, axis=1)
    return pivot_data

def day_sequencing_interpolate(pivot_data, domain_ids, OBP):
    # 1) sorting 
    data = pivot_data.sort_values(['person_id', 'concept_date'], ascending=[True, True])
    
    # 2) index : concept date (remove index name)
    data = data.set_index('concept_date', drop=False)
    data = data.rename_axis(None, axis=0)
    
    # 3) create empty dataframe
    df_col = pd.DataFrame(columns=data.columns)

    for person_id, group_df in data.groupby(['person_id']):
        
        # 4) less than 7 days of data
        train_min = pd.to_datetime( group_df['concept_date'].min() ) #last train date
        train_max = pd.to_datetime( group_df['concept_date'].max() ) #last train date
        first_abnormal_date_max = pd.to_datetime( group_df['first_abnormal_date'].max() ) #last train date
        
        #print(train_min, train_max, train_max - train_min) 
        if train_min == train_max :
            # 4-1) only first abnormal date
            #print(person_id, train_min, train_max, first_abnormal_date_max)
            pass
        elif train_min > train_max-timedelta(days=OBP-1): # ex. 2/28 - 7day = 2/21
            # 4-1) create data index
            #group_df.loc[train_max-timedelta(days=OBP-1)] = None
            group_df.loc[train_max-timedelta(days=OBP)] = None

        # 5) day interpolating 
        temp_df = group_df.sort_index().asfreq('D', method = None)
        
        # 6) fill value common columns
        common_columns = ['person_id', 'sex', 'age', 'cohort_start_date', 'concept_date', 'first_abnormal_date', 'label']
        common_columns_value = temp_df[common_columns].mode().loc[0].to_dict()
        temp_df[common_columns] = temp_df[common_columns].fillna(common_columns_value)
        
        # 7) filling missing value
        meas_columns = list(set(domain_ids['meas']) & set(data.columns))
        temp_df[meas_columns] = temp_df[meas_columns].fillna(method='ffill').fillna(0)
        
        drug_columns = list(set(domain_ids['drug']) & set(data.columns))
        temp_df[drug_columns] = temp_df[drug_columns].fillna(0)
        
        proc_columns = list(set(domain_ids['proc']) & set(data.columns))
        temp_df[proc_columns] = temp_df[proc_columns].fillna(0)
        
        cond_columns = list(set(domain_ids['cond']) & set(data.columns))
        temp_df[cond_columns] = temp_df[cond_columns].fillna(method='ffill').fillna(0)
        
        # 7) add df 
        df_col = pd.concat([df_col, temp_df], sort=False)
        
    # 8) set concept_date
    df_col['concept_date'] = df_col.index
    return df_col

def shift_rolling_window(input_df, OBP, nShift, uid_index):
    # OBP = 28
    # nShift = 7
    # cnt = 1
    # 1) create empty dataframe
    new_columns = input_df.columns.insert(0, 'unique_id')
    output_df = pd.DataFrame(columns=new_columns)
    # 2) slice data by subject id
    for person_id, group_df in input_df.groupby(['person_id']):
        # 3) slice data by shift n days (maximun 6 months)
        for i in range(1, 180+1, nShift):
            slice_df = group_df.shift(i).tail(OBP)
            # 4) if has nan data break.
            if slice_df.isnull().values.any():
                break
            # 5) if label 1 only in the last 4 weeks
            # if i != 1:
            #     slice_df['label'] = 0
            # 6) slice index
            slice_df['unique_id']=uid_index
            # 7) set index
            slice_df = slice_df.set_index('concept_date', drop=False)
            # 8) add slice data to data frame
            output_df = pd.concat([output_df, slice_df], sort=False)
            uid_index += 1
            # print(slice_df)
    # 9) sorting 
    output_df = output_df.sort_values(['unique_id','concept_date'])
    print(uid_index)
    return output_df

def label_0_fitting(input_df, OBP, nShift, uid_index):
    # OBP = 28
    # nShift = 7
    # 1) create empty dataframe
    new_columns = input_df.columns.insert(0, 'unique_id')
    output_df = pd.DataFrame(columns=new_columns)
    # 2) slice data by subject id
    for person_id, group_df in input_df.groupby(['person_id']):
        # 3) slice data by shift n days (maximun 6 months)
        for i in range(0, 60, nShift):
            slice_df = group_df.shift(i).tail(OBP)
            # 4) if has nan data break.
            if OBP != len(slice_df):
                break
            if slice_df.isnull().values.any():
                break
            # 6) slice index
            slice_df['unique_id'] = uid_index
            # 7) set index
            slice_df = slice_df.set_index('concept_date', drop=False)
            # 8) add slice data to data frame
            output_df = pd.concat([output_df, slice_df], sort=False)
            uid_index += 1
            # print(slice_df)
    # 9) sorting 
    output_df = output_df.sort_values(['unique_id','concept_date'])
    print(uid_index)
    return output_df

# OBP = 28
# nShift = 7
# #temp_df = label_1.query("person_id==5000635")
# temp_df = label_1.query("person_id==309154")

# train_min = pd.to_datetime( temp_df['concept_date'].min() ) #last train date
# train_max = pd.to_datetime( temp_df['concept_date'].max() ) #last train date
# temp_df['concept_date']
# print(train_min, train_max, (train_max-train_min).days)

# df_col = pd.DataFrame(columns=temp_df.columns)
# diff_date = (train_max-train_min).days
# #63-28-7-7-7-7
# # nCount = int((diff_date-OBP+1)/nShift)+1
# print(nCount)
# for i in range(nCount):
#     print(i)
#  #df.isnull().values.any()
# #print(diff_date.days)
# #for i in range():
    
# #print(train_max, min_date)
# # while i < 6:
    
# #     temp_df2 = temp_df.shift(nShift).tail(OBP)
# #     temp_df2
# #     df_col = pd.concat([df_col, temp_df], sort=False)
# # temp_df2 = temp_df.shift(periods=7)
# # temp_df2

def normalization(df):
    def min_max_scaler(df):
        normalized_df = (df-df.min())/(df.max()-df.min())
        return normalized_df
    def average_normal(df):
        normalized_df=(df-df.mean())/df.std()
        return normalized_df
    # 'unique_id', 'person_id', 'sex', 'cohort_start_date', 'concept_date', 'first_abnormal_date', 'label'
    target_columns = df.columns.difference(['unique_id', 'person_id', 'sex', 'cohort_start_date', 'concept_date', 'first_abnormal_date', 'label'])
    df[target_columns] = min_max_scaler(df[target_columns])
    return df

def get_matching_pairs(label1_df, label0_df, scaler=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    
    label1_x = label1_df.values.reshape(-1,1)
    label0_x = label0_df.values.reshape(-1,1)
    
    if scaler == True:
        scaler = StandardScaler()
    
    if scaler:
        scaler.fit(label1_x)
        label1_x = scaler.transform(label1_x)
        label0_x = scaler.transform(label0_x)
        
    nbrs = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(label0_x)
    distance, indices = nbrs.kneighbors(label1_x)
    # print(indices.shape)
    indices = indices.reshape(indices.shape[0]*5)
    matched = label0_df.iloc[indices]
    return matched

def extract_past_data_based_on_index_date(x):
    return x.query('cohort_start_date >= concept_date')
    
def filter_with_missing_rate(x, nCaseInTotal, nControlInTotal, threshold):
    past_data = extract_past_data_based_on_index_date(x)
    nCaseInConceptId = len(past_data.loc[past_data['label']==1,'person_id'].unique())
    nControlInConceptId =len(past_data.loc[past_data['label']==0,'person_id'].unique())
    fEpsilon = 1.0e-08 # devide by zero
    fMissingRateForCase = nCaseInConceptId / (nCaseInTotal + fEpsilon)
    fMissingRateForControl = nControlInConceptId / (nControlInTotal + fEpsilon)
    if (fMissingRateForCase < threshold) | (fMissingRateForControl < threshold) :
        return pd.DataFrame(columns=past_data.columns)
    # print("{}, {}, {}, {}, {:.2}, {}, {}, {:.2}".format(set(past_data.concept_id), set(past_data.concept_name), nCaseInConceptId, nCaseInTotal, fMissingRateForCase, nControlInConceptId, nControlInTotal, fMissingRateForControl))
    return x

def filter_with_missing_rate_concept(x, nCaseInTotal, nControlInTotal, threshold):
    past_data = extract_past_data_based_on_index_date(x)
    nCaseInConceptId = len(past_data.loc[past_data['label']==1,'person_id'].unique())
    nControlInConceptId =len(past_data.loc[past_data['label']==0,'person_id'].unique())
    fEpsilon = 1.0e-08 # devide by zero
    fMissingRateForCase = nCaseInConceptId / (nCaseInTotal + fEpsilon)
    fMissingRateForControl = nControlInConceptId / (nControlInTotal + fEpsilon)
    if (fMissingRateForCase < threshold) | (fMissingRateForControl < threshold) :
        return pd.DataFrame(columns=past_data.columns)
    # print("{}, {}, {}, {}, {:.2}, {}, {}, {:.2}".format(set(past_data.concept_id), set(past_data.concept_name), nCaseInConceptId, nCaseInTotal, fMissingRateForCase, nControlInConceptId, nControlInTotal, fMissingRateForControl))
    
    temp_df = pd.DataFrame(columns = ['concept_id', 'concept_name', 'nCaseInConceptId', 'nCaseInTotal', 'fMissingRateForCase', 'nControlInConceptId', 'nControlInTotal', 'fMissingRateForControl'])
    var_temp = {}
    var_temp['concept_id'] = set(past_data.concept_id)
    var_temp['concept_name'] = set(past_data.concept_name)
    var_temp['nCaseInConceptId'] = nCaseInConceptId
    var_temp['nCaseInTotal'] = nCaseInTotal
    var_temp['fMissingRateForCase'] = fMissingRateForCase
    var_temp['nControlInConceptId'] = nControlInConceptId
    var_temp['nControlInTotal'] = nControlInTotal
    var_temp['fMissingRateForControl'] = fMissingRateForControl
    temp_df = temp_df.append(var_temp, ignore_index=True)
    return temp_df