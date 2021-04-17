# -*- coding: utf-8 -*-
from typing import Tuple
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import math

import sklearn

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score

from utils import timing

train_spreadsheet_id = '1OdjccfGlv3lsuiWgIAHbE8id91FpVaU2EsaZo5kknaA'
test_spreadsheet_id = '1RzcxaIM2nVAsmKydLR1NnjqdJlUC86SAUOeW_L0mJgk'
file_link = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'


def get_data(file_id: str) -> pd.DataFrame:
    r = requests.get(file_link.format(file_id))
    return pd.read_csv(BytesIO(r.content))


def dates_from_strings(df: pd.DataFrame) -> None:
    df.loc[:, 'Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], format='%Y-%m-%d')


def split_train_test_emps(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_test = get_data(test_spreadsheet_id)
    id_list = df_test['Emp_ID'].tolist()
    for_test = df['Emp_ID'].isin(id_list)
    return df[~for_test], df[for_test]


def generate_emp_samples(df):
    ids = df['Emp_ID'].unique()
    max_date = df['MMM-YY'].max()

    employee_dfs = [df[df['Emp_ID'] == i] for i in ids]
    employee_features = {}
    for employee_df in employee_dfs:
        id = employee_df.iloc[0]['Emp_ID']
        employee_features[id] = {}

        employee_features[id]['Emp_ID'] = id
        employee_features[id]['Salary Change'] = (employee_df['Salary'].max() - employee_df['Salary'].min()) / \
                                                 employee_df['Salary'].min()
        employee_features[id]['Total Business Value All'] = employee_df['Total Business Value'].sum()
        employee_features[id]['Overvalue'] = (employee_df['Total Business Value'] / employee_df['Salary']).mean()

        last_day = pd.Timestamp(employee_df.tail(1)['LastWorkingDate'].iloc[-1])
        if pd.isnull(last_day):
            last_day = float('NaN')
            last_working_day = max_date
        else:
            last_working_day = last_day

        # employee_features[id]['LastWorkingDate'] = last_day

        join_date = employee_df[employee_df['Emp_ID'] == id]['Dateofjoining'].iloc[0]
        # employee_features[id]['Dateofjoining'] = join_date

        # Work experience: for not-fired calculated at max_date
        employee_features[id]['Work Experience'] = math.ceil((last_working_day - join_date) / np.timedelta64(1, 'M'))

        employee_features[id]['Fired'] = not employee_df['LastWorkingDate'].isnull().values.all()

    return pd.DataFrame.from_dict(employee_features, orient='index')

def extract_employee_features(df):
    ids = set(df['Emp_ID'])
    eployee_dfs = [df[df['Emp_ID'] == i] for i in ids]

    employee_features = {}
    for employee_df in eployee_dfs:
        #- ----
        #employee_df = employee_df.iloc[-6:].copy()
        #-----
        
        id = employee_df.iloc[0]['Emp_ID']
        employee_features[id] = {}
        employee_features[id]['Salary Change'] = (employee_df['Salary'].max() - employee_df['Salary'].min()) / employee_df['Salary'].min()
        employee_features[id]['Salary Average'] = employee_df['Salary'].mean()
        employee_features[id]['Salary changed'] = employee_features[id]['Salary Change'] != 0
        employee_features[id]['Salary'] = employee_df.iloc[-1]['Salary']
        employee_features[id]['Total Business Value All'] = employee_df['Total Business Value'].sum()
        employee_features[id]['Overvalue'] = (employee_df['Total Business Value'] / employee_df['Salary']).mean()
        employee_features[id]['Fired'] = employee_df.iloc[0]['Fired']
        employee_features[id]['Age'] = employee_df.iloc[0]['Age']
        employee_features[id]['Emp_ID'] = employee_df.iloc[0]['Emp_ID']
        employee_features[id]['Quarterly Rating'] = employee_df.iloc[-1]['Quarterly Rating']
        employee_features[id]['Designation'] = employee_df.iloc[-1]['Designation']
        
        gender_mapping = {'Male': 0, 'Female': 1}
        
        employee_features[id]['Gender'] = gender_mapping[employee_df.iloc[-1]['Gender']]

        education_mapping = {'College': 1,'Bachelor':2, 'Master':3}

        employee_features[id]['Education_Level'] = education_mapping[employee_df.iloc[-1]['Education_Level']]


        last_day = pd.to_datetime(employee_df.iloc[-1]['MMM-YY'], format='%Y-%m-%d')
        first_day = pd.to_datetime(employee_df.iloc[0]['MMM-YY'], format='%Y-%m-%d')
        employee_features[id]['Work Experience'] = math.ceil((last_day - first_day) / np.timedelta64(1, 'M'))

        join_day = employee_df[employee_df['Emp_ID'] == id]['Dateofjoining'].iloc[0]
        employee_features[id]['Dateofjoining'] = join_day
    return pd.DataFrame.from_dict(employee_features, orient='index')
    

def filter_data(df, end_date):
    df = df[(df['MMM-YY'] <= end_date) & (df['Dateofjoining'] < end_date)].copy()
    df['LastWorkingDate'] = np.where(df['LastWorkingDate'] >= end_date, pd.NaT, df['LastWorkingDate'])
    return df

def filter_dataset(df, date):
    before = df[df['MMM-YY'] < date].copy()

    before['Fired'] = [False] * len(before)
    
    after = df[df['MMM-YY'] > date]
    year, month, *rest = date.split('-')
    year = int(year)
    month = int(month) + 6 - 1
    year += month // 12
    month = month % 12 + 1
    future_end_date = "%04i-%02i-00" % (year, month)
    after = after[after['MMM-YY'] < future_end_date].copy()
    
    ids_before = before['Emp_ID']
    ids_after = after['Emp_ID']
    intersection_id = list(set(ids_before) & set(ids_after))
    filtered = before[before['Emp_ID'].isin(intersection_id)].copy()
    for id in intersection_id:
        filtered.loc[filtered['Emp_ID'] == id, 'Fired'] = len(after[(after['Emp_ID'] == id) & ~(after['LastWorkingDate'].isnull())])
    return filtered

def month_year_iter( start_month, start_year, end_month, end_year ):
    ym_start= 12*start_year + start_month - 1
    ym_end= 12*end_year + end_month - 1
    for ym in range( ym_start, ym_end ):
        y, m = divmod( ym, 12 )
        yield y, m+1

def iterate_through_set(df, start_month, start_year, end_month, end_year):
    result = []
    for year, month in month_year_iter(start_month, start_year, end_month, end_year):
        print('.',end='')
        result.append(
            extract_employee_features(
                filter_dataset(df, "%04i-%02i-00" % (year, month))
            )
        )
    return pd.concat(result, ignore_index=True)
       

def main():
    bad_ids = [264, 1207, 1581, 2397, 66, 383, 612, 743, 755, 770, 920, 1173, 1224, 1339, 1437, 1449, 1454, 1561, 1629, 1663, 1706, 1758, 1893, 1894, 2132, 2268, 2547, 2685]

    with timing('Loading data'):
        full_df = get_data(train_spreadsheet_id)

    with timing('Processing data'):

        full_df['Fired'] = [False] * len(full_df)
        full_df = full_df[~full_df['Emp_ID'].isin(bad_ids)]

        filtered = pd.read_pickle('/home/syslink/Documents/Hackathon/filtered.pkl')

        #filtered = iterate_through_set(full_df, 2, 2016, 6, 2017)

        #filtered_fired = iterate_through_set(full_df, 2, 2016, 12, 2017)
        #filtered_fired = filtered_fired[filtered_fired['Fired'] == True]
        #filtered_not_fired = iterate_through_set(full_df, 2, 2016, 6, 2017)
        #filtered_not_fired = filtered_not_fired[filtered_not_fired['Fired'] == False]
        #filtered = pd.concat([filtered_fired, filtered_not_fired])

        filtered.to_pickle('/home/syslink/Documents/Hackathon/filtered.pkl')
        
        filtered = filtered.drop('Salary Change', axis=1)
        #filtered = filtered.drop('Salary Average', axis=1)
        filtered = filtered.drop('Salary changed', axis=1)
        filtered = filtered.drop('Salary', axis=1)
        filtered = filtered.drop('Total Business Value All', axis=1)
        filtered = filtered.drop('Overvalue', axis=1)
        #iltered = filtered.drop('Designation', axis=1)
        #filtered = filtered.drop('Age', axis=1)
        filtered = filtered.drop('Gender', axis=1)
        #filtered = filtered.drop('Education_Level', axis=1)
        #filtered = filtered.drop('Quarterly Rating', axis=1)
        #filtered = filtered.drop('Work Experience', axis=1)
        df_test = get_data(test_spreadsheet_id)
        id_list = df_test['Emp_ID'].tolist()
        # scaling data 

        sc = StandardScaler()
        features_to_scale = list(set(['Education_Level', 'Salary', 'Overvalue', 'Salary Change', 'Salary changed', 'Salary Average', 'Age', 'Quarterly Rating', 'Total Business Value All', 'Work Experience', 'Designation']) & set(list(filtered.columns)))
        filtered[features_to_scale] = sc.fit_transform(filtered[features_to_scale])

        X, val = split_train_test_emps(filtered)
        val = extract_employee_features(
                full_df[full_df['Emp_ID'].isin(id_list)]
            )

        val[features_to_scale] = sc.transform(val[features_to_scale])
        val = val[list(filtered.columns)]
        # removing not used for training data

        y = list(X['Fired'])
        print("Fired: ", len(X[X['Fired'] == True]) / len(X))
        X = X.drop('Fired', axis=1)
        X = X.drop('Emp_ID', axis=1)
        X = X.drop('Dateofjoining', axis=1)
        val = val.drop('Dateofjoining', axis=1)
        val = val.drop('Fired', axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234,
                                                                stratify=y)

    print("Employees in train before: ", len(X_train))
    print("Employees in test before: ", len(X_test))
    ros = RandomUnderSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    print("Employees in train after: ", len(X_train))
    print("Employees in test after: ", len(X_test))

    print("Fired percent employees in train: ", len([x for x in y_train if x == 1]) / len(X_train) * 100)  #4794
    print("Fired percent employees in test: ", len([x for x in y_test if x == 1]) / len(y_test) * 100)  #0

    #clf = XGBClassifier()
    #clf.fit(X_train, y_train) 
    #clf = RandomForestClassifier(warm_start=True, n_estimators=90) # 
    clf = MLPClassifier(hidden_layer_sizes=50, solver='adam', learning_rate='adaptive', shuffle=True, activation='relu', max_iter=300)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, pred)
    print(cm)
    # accuracy score
    print('Accuracy:', accuracy_score(y_test, pred))

    # precision
    precision = cm[1][1] / (cm[0][1] + cm[1][1])
    print('Precision:', precision)

    # recall
    recall = cm[1][1] / (cm[1][0] + cm[1][1])
    print('Recall:', recall)

    # F1_score
    print('F1:', f1_score(y_test, pred))
    print('F1 Train:', f1_score(clf.predict(X_train), y_train))

    # obtain prediction probabilities
    #pred = clf.predict_proba(X_test)
    #pred = [p[1] for p in pred]
    # AUROC score
    print('AUROC:', roc_auc_score(y_test, pred))


    with timing('Saving results'):
        final = clf.predict(val.drop('Emp_ID', axis=1))
        df_test = val[['Emp_ID']].copy()
        df_test['Target'] = final.astype(int)
        df_test.to_csv('output.csv', index=False)

    print('Distribution of target feature in test predictions:')
    print(df_test['Target'].value_counts())


if __name__ == '__main__':
    main()