import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm
from lightgbm import LGBMClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter('ignore')

## read data 

client_train = pd.read_csv('data/client_train.csv')
invoice_train = pd.read_csv('data/invoice_train.csv')
client_test = pd.read_csv('data/client_test.csv')
invoice_test = pd.read_csv('data/invoice_test.csv')


## deal with categorical features

# 'month'
invoice_train['invoice_date'] = pd.to_datetime(invoice_train['invoice_date'])
invoice_train['month'] = invoice_train['invoice_date'].dt.month

# 'counter_statue'
invoice_train['counter_statue'] = invoice_train['counter_statue'].astype(str)
filtered_values = ['0', '1', '2', '3', '4', '5']
invoice_train = invoice_train[invoice_train['counter_statue'].isin(filtered_values)]

# 'tariif_type', 'reading_remarque', 'counter_type', 'counter_number'

# all categorical features
cat_features = ['month', 'counter_statue', 'tarif_type', 'reading_remarque', 'counter_type', 'counter_number']

# one-hot encoding
for cat_feature in cat_features:
    encoder = OneHotEncoder()
    encoder.fit(invoice_train[cat_feature].values.reshape(-1, 1))
    tmp = encoder.transform(invoice_train[cat_feature].values.reshape(-1, 1)).toarray()
    tmp = pd.DataFrame(tmp, columns=[(cat_feature + '_' + str(i)) for i in range(tmp.shape[1])])
    invoice_train = pd.concat([invoice_train, tmp], axis=1)
    invoice_train.drop(cat_feature, axis=1, inplace=True)


## deal with numerical features

# 'coeficient'

# 'months_number'
invoice_train = invoice_train.query('months_number <= 24')

# 'old_index', 'new_index' -> 'consumption', 'consumption_ave'
invoice_train['consumption'] = invoice_train['new_index'] - invoice_train['old_index']
invoice_train['consumption_ave'] = invoice_train['consumption'] / invoice_train['months_number']
invoice_train = invoice_train.query('consumption >= 0')

# 'consommation_level_1','consommation_level_2', 'consommation_level_3', 'consommation_level_4'

# aggregate features by client_id
def aggregate_by_client_id(invoice_data, cat_features):

    aggs = {
        'consommation_level_1': ['mean'],
        'consommation_level_2': ['mean'],
        'consommation_level_3': ['mean'],
        'consommation_level_4': ['mean'],
        'months_number': ['max', 'min', 'mean'],
        'consumption': ['sum', 'max', 'min', 'mean'],
        'consumption_ave': ['max', 'min', 'mean']
    }

    for cat in cat_features:
        aggs[cat] = ['count', 'nunique']

    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = invoice_data.groupby('client_id').size().reset_index(name='transactions_count')
    
    return pd.merge(df, agg_trans, on='client_id', how='left')

invoice_train_aggregated = aggregate_by_client_id(invoice_train, cat_features)

print(invoice_train_aggregated.head())


'''
def aggregate_by_client_id(invoice_data):

    aggs = {}

    aggs['consommation_level_1'] = ['mean']
    aggs['consommation_level_2'] = ['mean']
    aggs['consommation_level_3'] = ['mean']
    aggs['consommation_level_4'] = ['mean']
    
    aggs['months_number'] = ['max', 'min', 'mean']
    aggs['consumption'] = ['sum', 'max', 'min', 'mean']
    aggs['consumption_ave'] = ['max', 'min', 'mean']

    # Perform aggregation by client_id
    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    
    # Rename the columns
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    # Count the number of transactions per client
    df = invoice_data.groupby('client_id').size().reset_index(name='transactions_count')
    
    # Merge the aggregated features and transaction count into one DataFrame
    return pd.merge(df, agg_trans, on='client_id', how='left')

invoice_train_aggregated = aggregate_by_client_id(invoice_train)
'''

