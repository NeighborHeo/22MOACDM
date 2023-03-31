import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

def save_JointPlot(df, filedir, filename):
    plt.figure()
    joint_plot = sns.jointplot(x="age", y="value_as_number", data=df, kind="kde", hue="gender_source_value", markers=["M", "F"])
    plt.suptitle(t="{}(age-value) plot".format(filename), y=1.02)
    plt.xlabel(xlabel="age")
    plt.ylabel(ylabel="lab value")
    plt.legend(loc="upper left")
    joint_plot.savefig('{}/{}_plot1.png'.format(filedir, filename), dpi=300)
    plt.show()
    
def save_quadplot(df, filedir, filename):
    plt.figure()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    #1 
    plot1 = axes[0,0]
    sns.countplot(ax=plot1, data=df, x="gender_source_value")
    plot1.set_title("count by gender")
    plot1.set_xlabel(xlabel="gender")
    plot1.set_ylabel(ylabel="count (n)")
    plot1.grid(True, lw = 0.5)

    #2
    plot2 = axes[0,1]
    sns.histplot(ax=plot2, data=df, x="value_as_number")
    plot2.set_title("count by value")
    plot2.set_xlabel(xlabel="value")
    plot2.set_ylabel(ylabel="count (n)")
    plot2.legend(loc='upper right')

    #2
    plot3 = axes[1,0]
    sns.lineplot(ax=plot3, x='age_dec', y="person_id",data=df[df['gender_source_value']=='M'].groupby(['age_dec']).count(), marker='o', alpha=0.6, label='M')
    sns.lineplot(ax=plot3, x='age_dec', y="person_id",data=df[df['gender_source_value']=='F'].groupby(['age_dec']).count(), marker='o', alpha=0.6, label='F')
    sns.lineplot(ax=plot3, x='age_dec', y="person_id", data=df.groupby(['age_dec']).count(), marker='o', label='ALL')
    plot3.set_title("count by age")
    plot3.set_xlabel(xlabel="age")
    plot3.set_ylabel(ylabel="count (n)")
    plot3.grid(True, lw = 0.5)#그래프 보조선을 그음, lw - 선의 굵기 설정

    #3
    plot4 = axes[1,1]
    ax = sns.boxplot(ax=plot4, x="age_dec", y="value_as_number", data=df, hue="gender_source_value") #, palette="Set3"
    plot4.set_title("value by age and gender")
    plot4.set_xlabel(xlabel="age")
    plot4.set_ylabel(ylabel="value")
    plot4.legend(loc='upper right')

    fig.savefig('{}/{}_plot2.png'.format(filedir, filename), dpi=300)

def bin_by(x, y, nbins=30, bins = None):
    """
    Divide the x axis into sections and return groups of y based on its x value
    """
    if bins is None:
        bins = np.linspace(x.min(), x.max(), nbins)
    bin_space = (bins[-1] - bins[0])/(len(bins)-1)/2
    indicies = np.digitize(x, bins + bin_space)
    output = []
    for i in range(0, len(bins)):
        output.append(y[indicies==i])
    # prepare a dataframe with cols: median; mean; 1up, 1dn, 2up, 2dn, 3up, 3dn
    df_names = ['mean', 'median', '5th', '95th', '10th', '90th', '25th', '75th']
    df = pd.DataFrame(columns = df_names)
    to_delete = []
    # for each bin, determine the std ranges
    for y_set in output:
        if y_set.size > 0:
            av = y_set.mean()
            intervals = np.percentile(y_set, q = [50, 5, 95, 10, 90, 25, 75])
            res = [av] + list(intervals)
            df = df.append(pd.DataFrame([res], columns = df_names))
        else:
            # just in case there are no elements in the bin
            to_delete.append(len(df) + 1 + len(to_delete))
    # add x values
    bins = np.delete(bins, to_delete)
    df['x'] = bins
    return df

def save_percentile_plot(df, filedir, filename):
    x = df['age'].to_numpy()
    y = df['value_as_number'].to_numpy()
    idx   = np.argsort(x)
    x = np.array(x)[idx]
    y = np.array(y)[idx]

    # bin the values and determine the envelopes
    df = bin_by(x, y, nbins=25, bins = None)

    # ###
    # # Plot 1
    # ###
    # # determine the colors
    # cols = ['#EE7550', '#F19463', '#F6B176']

    # with plt.style.context('fivethirtyeight'): 
    #     # plot the 3rd stdv
    #     plt.fill_between(df.x, df['5th'], df['95th'], alpha=0.7,color = cols[2], label="5th/95th")
    #     plt.fill_between(df.x, df['10th'], df['90th'], alpha=0.7,color = cols[1], label="10th/90th")
    #     plt.fill_between(df.x, df['25th'], df['75th'], alpha=0.7,color = cols[0], label="25th/75th")
    #     # plt the line
    #     plt.plot(df.x, df['median'], color = '1', alpha = 0.7, linewidth = 1)
    #     # plot the points
    #     plt.scatter(x, y, facecolors='white', edgecolors='0', s = 5, lw = 0.7, alpha=0.2, label = 'data')
    #     plt.legend()

    # plt.savefig('fig1.png', facecolor='white', edgecolor='none')
    # plt.show()

    # ###
    # # Plot 2 - same color, dif transparencies
    # ###
    # # determine the color
    # col = list(plt.style.library['fivethirtyeight']['axes.prop_cycle'])[0]['color']

    # with plt.style.context('fivethirtyeight'): 
    #     # plot the 1st band
    #     plt.fill_between(df.x, df['5th'], df['95th'], alpha=0.2,color = col)
    #     # plot the 2nd band
    #     plt.fill_between(df.x, df['10th'], df['90th'], alpha=0.6,color = col)
    #     plt.fill_between(df.x, df['25th'], df['75th'], alpha=1,color = col)
    #     # plt the line
    #     plt.plot(df.x, df['median'], color = '1', alpha = 0.7, linewidth = 1)
    #     # plot the points
    #     plt.scatter(x, y, facecolors='white', edgecolors='0', s = 5, lw = 0.7)

    # plt.savefig('fig2.png', facecolor='white', edgecolor='none')
    # plt.show()
    
    ###
    # Plot 3 - same color, dif transparencies
    ###
    # determine the color
    plt.figure()
    col1 = list(plt.style.library['fivethirtyeight']['axes.prop_cycle'])[0]['color']
    col2 = list(plt.style.library['fivethirtyeight']['axes.prop_cycle'])[1]['color']
    fig_width = 30#15.24#6.5 # if columns==1 else 6.9 # width in cm
    font_size = 9
    #
    golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_height = fig_width*golden_mean # height in cm,
    params = {
            'axes.labelsize': font_size, # fontsize for x and y labels (was 10)
            'font.size': font_size, # was 10
            'legend.fontsize': font_size, # was 10
            'xtick.labelsize': font_size,
            # 'ytick.labelsize': 0,
            'lines.linewidth' : 2,
            'figure.autolayout' : True,
            'figure.figsize': [fig_width/2.54,fig_height/2.54]
            }
        
    with plt.style.context('fivethirtyeight'): 
        plt.rcParams.update(params)
        # plot the points
        # plt.scatter(x, y, facecolors=col1)
        # plot the 1st band
        plt.fill_between(df.x, df['5th'], df['95th'], alpha=0.1,color = col2, label="5th/95th")
        # plot the 2nd band
        plt.fill_between(df.x, df['10th'], df['90th'], alpha=0.5,color = col2, label="10th/90th")
        plt.fill_between(df.x, df['25th'], df['75th'], alpha=0.9,color = col2, label="25th/75th")
        
        plt.scatter(x, y, facecolors='white', edgecolors='0', s = 5, lw = 0.7, alpha=0.2, label = 'data')
        # plt the line
        plt.plot(df.x, df['median'], color = '1', alpha = 0.7, linewidth = 1)
        plt.legend()

    plt.savefig('{}/{}_plot3.png'.format(filedir, filename), dpi=300, facecolor='white', edgecolor='none')
    plt.show()