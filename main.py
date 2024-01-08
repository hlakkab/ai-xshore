import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error,  r2_score

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


df1 = pd.read_csv('C:/Users/LENOVO/Downloads/sf.csv')
df1['date'] = pd.to_datetime(df1['date'])

for i in range(3, 9):
    df1.loc[:, 'date'] = df1['date'].apply(lambda x: x.replace(month=i))
    if 'df_4' in locals():
        df_4 = pd.concat([df_4, df1], ignore_index=True)
        print('2')
    else:
        df_4 = df1
        print('1')

df_4

df5 = df_4.set_index('date')
df5.index = pd.to_datetime(df5.index)




df6 = create_features(df5)

train = df6.loc[df6.index < '2023-08-27']
test = df6.loc[df6.index >= '2023-08-30']
df_30 = df6[df6.index > '2023-08-31']

print(df6.head(2))



FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'count'

X_train = train[FEATURES]
y_train = train[TARGET]

print(X_train.head(2))


X_test = test[FEATURES]
y_test = test[TARGET]

df_31 = df_30[["dayofyear","hour","dayofweek","quarter","month","year"]]

def train_the_model(X_train, y_train, X_test, y_test):
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    return reg

reg = train_the_model(X_train, y_train, X_test, y_test)
test['prediction'] = reg.predict(X_test)
max_values_index = test.sort_values(by='prediction',ascending = False)
top_5 = max_values_index.head(5)

print('----------')
print(top_5['prediction'])
print("---------------")