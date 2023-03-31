import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import os
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from datetime import timedelta
from sklearn import preprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import warnings
warnings.filterwarnings(action='ignore')

def readfile(filepath):
    f = open(filepath, 'r')
    text = f.read()
    f.close()
    return text

def cohortConditionSetting(domain_df, pre_observation_period, post_observation_peroid, delDataAfterAbnormal=True):
    """@ """
    from datetime import timedelta
    prev_len = len(domain_df)
    domain_df['cohort_start_date'] = pd.to_datetime(domain_df['cohort_start_date'], infer_datetime_format=True)
    domain_df['first_abnormal_date'] = pd.to_datetime(domain_df['first_abnormal_date'], infer_datetime_format=True)
    domain_df['concept_date'] = pd.to_datetime(domain_df['concept_date'], infer_datetime_format=True)
    # condition 1) Select patients with first adverse events within 2 months of cohort initiation.
    domain_df = domain_df[(domain_df['cohort_start_date']<=domain_df['concept_date']+timedelta(days=pre_observation_period))]
    # condition 2) Delete data before the cohort start date.
    domain_df = domain_df[(domain_df['concept_date']<=domain_df['cohort_start_date']+timedelta(days=post_observation_peroid))]
    # condition 3) Delete data after first_abnormal_date (Except when there is no first abnormal date.)
    if delDataAfterAbnormal:
        domain_df = domain_df[~(domain_df['first_abnormal_date']<domain_df['concept_date'])]
    domain_df.reset_index(drop=True, inplace=True)
    curr_len = len(domain_df)
    print('{} > {}'.format(prev_len, curr_len))
    return domain_df

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
