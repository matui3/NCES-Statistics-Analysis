import cse163_utils  # noqa: F401
# This is a hacky workaround to an unfortunate bug on macs that causes an
# issue with seaborn, the graphing library we want you to use on this
# assignment.  You can just ignore the above line or delete it if you are
# not using a mac!
# Add your imports and code here!

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def plot_tree(model, X, y):
  dot_data = export_graphviz(model, out_file=None,
                             feature_names=X.columns,
                             class_names=y.unique(),
                             filled=True, rounded=True,
                             special_characters=True)
  return graphviz.Source(dot_data)

def compare_bachelors_1980(data):
    df = data[(data['Min degree'] != 'high school') & (data['Min degree'] != 'associate\'s')]
    df = df[df['Year'] == 1980]
    df = df[df['Sex'] != 'A']
    df = df[['Sex', 'Total']]
    return df

def top_2_2000s(data):
    # Series = df.groupby(['Min degree'])['Total'].count()
    df = data[(data['Year'] >= 2000) & (data['Year'] <= 2010)].copy()
    df = df[['Min degree', 'Total']]
    df['Total'] = df['Total'].astype(float)
    df = df.groupby('Min degree')['Total'].mean()
    df = pd.Series.sort_values(df, ascending=False)
    df = df.drop(labels=['bachelor\'s', 'master\'s'])
    return df


def line_plot_bachelors(data):
    df = data[(data['Sex'] == 'A') & (data['Min degree'] == 'bachelor\'s')]
    df = df[['Year', 'Total']]
    df = df[df['Total'] != '---']
    df['Total'] = df['Total'].astype(float)
    sns.set_theme()
    g = sns.relplot(data = df, x = 'Year', y = 'Total', kind='line').set(title = "Percentage Earning Bachelor's over Time")
    g.set_axis_labels("Year", "Percentage")
    g.set(xlim = (1940, 2020), ylim = (5,35), xticks = [1940, 1960, 1980, 2000, 2020], yticks = [5, 10, 15, 20, 25, 30, 35] )
    plt.savefig('line_plot_bachelors.png', bbox_inches='tight')

def bar_chart_high_school(data):
    df = data[(data['Year'] == 2009)  (data['Min degree'] == 'high school')]
    df = df[['Sex', 'Total']]
    df = df[df['Total'] != '---']
    df['Total'] = df['Total'].astype(float)
    sns.set_theme()
    g = sns.catplot(data = df, x = 'Sex',  y = 'Total', kind = 'bar').set(title = "Percentage Completed High School by Sex")
    g.set_axis_labels("Sex", "Percentage")
    plt.savefig('bar_chart_high_school.png', bbox_inches='tight')

def plot_hispanic_min_degree(data):
    df = data[(data['Year'] >= 1990) & (data['Year'] <= 2010)]
    df = df[(df['Min degree'] == 'high school') | (df['Min degree'] == 'bachelor\'s')]
    df = df[df['Sex'] == 'A']
    df = df[['Min degree', 'Year', 'Hispanic']]
    df['Hispanic'] = df['Hispanic'].astype(float)
    sns.set_theme()
    g =  sns.relplot(data = df, x = 'Year', y = 'Hispanic', hue='Min degree', kind = 'line').set(title = "Percent of Hispanic Educational Attainment")
    g.set_axis_labels("Year", "Percentage")
    g.set(xlim = (1990, 2010), xticks = [1990, 1995, 2000, 2005, 2010])
    plt.savefig('plot_hispanic_min_degree.png', bbox_inches='tight')

def fit_and_predict_degrees(data):
    df = data[['Year', 'Min degree', 'Sex', 'Total']]
    df = df[df['Total'] != '---']
    model = DecisionTreeRegressor()
    X =  df.loc[:, df.columns != 'Total']
    X = pd.get_dummies(X)
    y = df['Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model.fit(X_train, y_train)
    accuracy_score(y_test, model.predict(X_test))
    

def main():
    df = pd.read_csv('hw3/hw3-nces-ed-attainment.csv')
    # compare_bachelors_1980(df)
    # top_2_2000s(df)
    # line_plot_bachelors(df)
    # bar_chart_high_school(df)
    # plot_hispanic_min_degree(df)
    fit_and_predict_degrees(df)



if __name__ == '__main__':
    main()
