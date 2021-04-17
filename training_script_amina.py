# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import math

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score

from utils import timing

train_spreadsheet_id = '1OdjccfGlv3lsuiWgIAHbE8id91FpVaU2EsaZo5kknaA'
test_spreadsheet_id = '1RzcxaIM2nVAsmKydLR1NnjqdJlUC86SAUOeW_L0mJgk'
file_link = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'


def get_data(file_id: str) -> pd.DataFrame:
    r = requests.get(file_link.format(file_id))
    return pd.read_csv(BytesIO(r.content))


def dates_from_strings(df: pd.DataFrame) -> None:
    df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], format='%Y-%m-%d')
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], format='%Y-%m-%d')
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], format='%Y-%m-%d')


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

        last_day = employee_df.tail(1)['LastWorkingDate'].iloc[-1]
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


def get_fitted_models(X_train, X_test, y_train, y_test):
    pipelines = {
        'l1': make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l1', random_state=123, solver='liblinear')),
        'l2': make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', random_state=123)),
        'rf': make_pipeline(
            RandomForestClassifier(random_state=123)),
        'gb': make_pipeline(
            GradientBoostingClassifier(random_state=123))
    }

    l1_hyperparameters = {'logisticregression__C': [0.001, 0.005, 0.01,
                                                    0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
                          }
    l2_hyperparameters = {'logisticregression__C':
                              [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                               1, 5, 10, 50, 100, 500, 1000]
                          }

    rf_hyperparameters = {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33],
        'randomforestclassifier__min_samples_leaf': [1, 3, 5, 10]
    }

    gb_hyperparameters = {
        'gradientboostingclassifier__n_estimators': [100, 200],
        'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
        'gradientboostingclassifier__max_depth': [1, 3, 5]
    }

    hyperparameters = {
        'l1': l1_hyperparameters,
        'l2': l2_hyperparameters,
        'rf': rf_hyperparameters,
        'gb': gb_hyperparameters
    }

    fitted_models = {}
    for name, pipeline in pipelines.items():
        model = GridSearchCV(pipeline,
                             hyperparameters[name],
                             cv=10,
                             n_jobs=-1)
        model.fit(X_train, y_train)
        fitted_models[name] = model

    for name, model in fitted_models.items():
        print(name, model.best_score_)
        print('Results for:', name)

        # obtain predictions
        pred = fitted_models[name].predict(X_test)
        # confusion matrix
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
        print('F1:', f1_score(y_test, [True, ] * len(y_test)))

        # obtain prediction probabilities
        pred = fitted_models[name].predict_proba(X_test)
        pred = [p[1] for p in pred]
        # plot ROC
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.plot(fpr, tpr, label=name)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('True Positive Rate (TPR) i.e. Recall')
        plt.xlabel('False Positive Rate (FPR)')
        plt.show()

        # AUROC score
        print('AUROC:', roc_auc_score(y_test, pred))
    return fitted_models


def main():
    with timing('Loading data'):
        full_df = get_data(train_spreadsheet_id)

    with timing('Processing data'):
        dates_from_strings(full_df)
        emp_full_df = generate_emp_samples(full_df)

    with timing('Dividing to train and test datasets'):
        train_df, test_df = split_train_test_emps(emp_full_df)

    y = train_df['Fired']
    X = train_df.drop('Fired', axis=1)

    print('Distribution of target feature in training dataset:')
    print(y.value_counts())

    with timing('Training'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234,
                                                            stratify=y)
        fitted_models = get_fitted_models(X_train, X_test, y_train, y_test)
        name = 'rf'

    with timing('Saving results'):
        final = fitted_models[name].predict(test_df.drop('Fired', axis=1))
        df_test = test_df[['Emp_ID']].copy()
        df_test['Target'] = final.astype(int)
        df_test.to_csv('output.csv', index=False)

    print('Distribution of target feature in test predictions:')
    print(df_test['Target'].value_counts())


if __name__ == '__main__':
    main()