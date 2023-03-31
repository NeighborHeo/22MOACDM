from re import M
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats 


def check_number_of_psmatches(df, treatments, nTimes):
    """
    df : dataframe
    treatments : list of treatment columns
    -> n_matches : number of matches
    """
    nCase = len(df[df[treatments] == 1])
    nControl = len(df[df[treatments] == 0])
    if (nCase==0) or (nCase*nTimes > nControl):
        return False
    return True

def union_set(set1, set2):
    return set(set1) | set(set2)

def get_matching_pairs(df, treatments, covariates, scaler=True):
    """
    df : dataframe
    treatments : list of treatment columns
    covariates : list of covariate columns
    scaler : bool
    -> matching_pairs : set indices of matching pairs
    """
    matched_df = pd.DataFrame()
    if False == check_number_of_psmatches(df, treatments, nTimes=1):
        matched_df = df[df[treatments] == 0]
        return matched_df
    
    set_index = False
    if not "idx" in df.columns:
        df["idx"] = df.index
        set_index = True
    
    case_df = df[df[treatments] == 1].reset_index(drop=True)
    control_df = df[df[treatments] == 0].reset_index(drop=True)
    case_x = case_df[covariates].values
    control_x = control_df[covariates].values

    if scaler == True:
        scaler = StandardScaler()

    if scaler:
        scaler.fit(case_x)
        case_x = scaler.transform(case_x)
        control_x = scaler.transform(control_x)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_x)
    distances, indices = nbrs.kneighbors(case_x)
    indices = indices.reshape(indices.shape[0])
    matched_sub_df = control_df.iloc[indices].reset_index(drop=True)
    matched_df = pd.concat([matched_df, matched_sub_df], axis=0)
    # If there is duplicate data in the matched data, it is executed again as a recursive function
    duplicated_matched_df = matched_sub_df[matched_sub_df.duplicated()]
    if (len(duplicated_matched_df) > 0):
        case_df = case_df[case_df.index.isin(duplicated_matched_df.index)]
        control_df = control_df[~control_df.idx.isin(matched_df.idx)]
        concat_df = pd.concat([case_df, control_df], axis = 0)
        matched_sub_df = get_matching_pairs(concat_df, treatments, covariates, scaler=True)
        matched_df = pd.concat([matched_df, matched_sub_df], axis=0)
    
    if set_index:
        matched_df.set_index("idx", inplace=True)
        
    return matched_df.drop_duplicates()

def get_matching_multiple_pairs(df, treatments, covariates, scaler=True, n_neighbors=3):
    """
    df : dataframe
    treatments : list of treatment columns
    covariates : list of covariate columns
    scaler : bool
    n_neighbors : int
    -> matching_pairs : set indices of matching pairs
    """
    df = df.copy()
    df['idx'] = df.index
    matched_df = pd.DataFrame()
    matched_df['idx'] = matched_df.index
    case_df = df[df[treatments] == 1]
    if False == check_number_of_psmatches(df, treatments, nTimes=n_neighbors):
        print(f"""The number of control groups compared to the case group is greater than {n_neighbors} times""")
        matched_df = df[df[treatments] == 0]
        concat_df = pd.concat([case_df, matched_df], axis=0)
        concat_df.set_index('idx', inplace=True)
        return concat_df

    for i in range(n_neighbors):
        df = df[~df.idx.isin(matched_df.idx)]
        matched_sub_df = get_matching_pairs(df, treatments, covariates, scaler)
        matched_df = pd.concat([matched_df, matched_sub_df], axis = 0)
        
    matched_df.drop_duplicates(inplace=True)
    matched_df.set_index('idx', inplace=True)
    concat_df = pd.concat([case_df, matched_df], axis=0)
    concat_df.set_index('idx', inplace=True)
    return concat_df

def generateData1(m, s, c, binary=False):
    """
    m = mean
    s = std
    c = count
    -> d : data
    """
    d = np.random.normal(m, s, c)
    if binary:
        d = np.where(d > 0.5, 1, 0)
    return d
    
def generateData2(m1, s1, c1, m2, s2, c2, binary=False):
    d1 = generateData1(m1, s1, c1, binary)
    d2 = generateData1(m2, s2, c2, binary)
    return np.concatenate([d1, d2], axis=0)

def make_test_data(nCase, nControl):
    data_dict = {}
    data_dict['x1'] = x1 = generateData2(57, 16, nCase, 56, 14, nControl)
    data_dict['x2'] = x2 = generateData2(0.75, 0.43, nCase, 0.64, 0.47, nControl, binary=True)
    data_dict['x3'] = x3 = generateData2(0.65, 0.3, nCase, 0.68, 0.39, nControl)
    data_dict['x4'] = x4 = generateData2(1, 0.1, nCase, 1, 0.1, nControl, binary=True)
    data_dict['y'] = y = generateData2(1, 0, nCase, 0, 0, nControl, binary=True)
    df = pd.DataFrame(data=data_dict)
    return df

def test():
    df = make_test_data(100, 1000)
    treatments = 'y'
    covariates = ['x1', 'x2', 'x3', 'x4']
    matched_df = get_matching_multiple_pairs(df, treatments, covariates, scaler=True, nTimes=3)
    print(len(set(matched_df.index)))

 
def user_shapiro_test(df, covariates):
    _, p = stats.shapiro(df.loc[:, covariates])
    print(f'p={p:.3f}')
    # interpret
    alpha = 0.05  # significance level
    print(f"covariates: {covariates} is normal distribution: {p:.3f} > {alpha}, {p > alpha}")
    if p > alpha:
        print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
    else:
        print('different distributions/different group mean (reject H0)')
    return p > alpha

def user_levene_test(df, target, covariates):
    _, p = stats.levene(df.loc[df[target] == 0, covariates], df.loc[df[target] == 1, covariates])
    print(f'p={p:.3f}')
    # interpret
    alpha = 0.05  # significance level
    print(f"covariates: {covariates} is equal variance: {p:.3f} > {alpha}, {p > alpha}")
    if p > alpha:
        print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
    else:
        print('different distributions/different group mean (reject H0)')
    return p > alpha

def user_t_test_ind(df, target, covariates):
    # separate control and treatment for t-test
    df_control = df[df[target]==0]
    df_case = df[df[target]==1]
    # compare samples
    user_shapiro_test(df_control, covariates)
    user_shapiro_test(df_case, covariates)
    
    equal_var = user_levene_test(df, target, covariates)
    _, p = stats.ttest_ind(df_control.loc[:, covariates], df_case.loc[:, covariates], equal_var=equal_var)
    print(f"case \t\t mean and std: {df_case[covariates].mean():.3f}, {df_case[covariates].std():.3f}")
    print(f"control \t mean and std: {df_control[covariates].mean():.3f}, {df_control[covariates].std():.3f}")
          
    # interpret
    alpha = 0.05  # significance level
    print(f"***covariates: {covariates} is average difference: {p:.3f} > {alpha}, {p > alpha}")
    if p > alpha : 
        print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
    else:
        print('different distributions/different group mean (reject H0)')
    return p

def user_mcnemar_test(df, target, covariates):
    from statsmodels.stats.contingency_tables import mcnemar
    # separate control and treatment for t-test
    df_control = df[df[target]==0]
    df_case = df[df[target]==1]
    
    table = np.vstack((df_control[covariates].value_counts(), df_case[covariates].value_counts()))
    
    print(table)
    table = np.transpose(table)             # trans pose
    result = mcnemar(table, exact=True, correction=True)
    # _, p = mcnemar(df_control.loc[:, covariates], df_case.loc[:, covariates])
    
    # _, p = stats.meneamar(df_control.loc[:, covariates], df_case.loc[:, covariates])
    # _, p = stats.chisquare(df_control.loc[:, covariates], df_case.loc[:, covariates])
    print(f"case \t\t mean and std: {df_case[covariates].mean():.3f}, {df_case[covariates].std():.3f}")
    print(f"control \t mean and std: {df_control[covariates].mean():.3f}, {df_control[covariates].std():.3f}")
          
    # interpret
    alpha = 0.05  # significance level
    print(f"***covariates: {covariates} is average difference: {result.pvalue:.3f} > {alpha}, {result.pvalue > alpha}")
    if result.pvalue > alpha : 
        print('same distributions/same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
    else:
        print('different distributions/different group mean (reject H0)')
    return result.pvalue

def check_covariate(df, target, covariates):
    binary = df[covariates].nunique() == 2
    if binary:
        return user_mcnemar_test(df, target, covariates)
    else:
        return user_t_test_ind(df, target, covariates)
