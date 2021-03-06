# data preprocess
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import xgboost as xgb

def read_csv_to_df(filename):
    df = pd.read_csv(filename)
    return df

def write_df_to_csv(df, filename):
    df.to_csv(filename, index=False)

def get_dmatrix(tr_filename, va_filename):
    dtr = xgb.DMatrix(tr_filename)
    dva = xgb.DMatrix(va_filename)
    return dtr, dva

def replace_NY_to_01(df, columns):
    for column in columns:
      df[column].replace('N', 0, inplace=True)
      df[column].replace('Y', 1, inplace=True)
    return df

def normalize_regression_feature(df, columns):
    for column in columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def one_hot_encoding(df, columns):
    for column in columns:
        df = pd.concat([df,pd.get_dummies(df[column], prefix=column)],axis=1)
        df.drop([column], axis=1, inplace=True)
    return df

def time_interval(df, column_intervals):
    for column, interval in column_intervals:
        bins_range = list(range(int(df[column].min()), int(df[column].max()+2), interval))
        label_range = list(range(0, len(bins_range)))
        bins_range.append(np.inf)
        df[column] = pd.cut(df[column], bins=bins_range, labels=label_range)
        df = one_hot_encoding(df, [column])
    return df

def _get_rows_based_on_column_val(df, col_name, val):
    rows = df.loc[df[col_name] == val]
    print('Rows number about {} with {}: {}'.format(col_name, val, rows.shape[0]))
    return rows

def get_fraud(df):
    fraud = _get_rows_based_on_column_val(df, 'fraud_ind', 1)
    return fraud

def get_real(df):
    real = _get_rows_based_on_column_val(df, 'fraud_ind', 0)
    return real

def preprocess_data1(df):
    # binary class -> N to 0, Y to 1
    df = replace_NY_to_01(df, ['ecfg', 'insfg', 'ovrlt', 'flbmk', 'flg_3dsmk'])
    # continuous value -> normalize value to 0~1
    df = normalize_regression_feature(df, ['conam', 'iterm'])
    df = one_hot_encoding(df, ['contp', 'etymd', 'stscd', 'hcefg', 'csmcu'])
    df.drop(['acqic', 'bacno', 'txkey', 'cano'], axis=1, inplace=True)
    return df

def preprocess_data2(df):
    # binary class -> N to 0, Y to 1
    df = replace_NY_to_01(df, ['ecfg', 'insfg', 'ovrlt', 'flbmk', 'flg_3dsmk'])
    # continuous value -> normalize value to 0~1
    df = normalize_regression_feature(df, ['conam', 'iterm'])
    df = one_hot_encoding(df, ['contp', 'etymd', 'stocn', 'stscd', 'hcefg', 'csmcu', 'locdt', 'mcc'])
    df = time_interval(df, [('loctm', 6380)])
    df.drop(['acqic', 'bacno', 'txkey', 'cano', 'mchno', 'scity'], axis=1, inplace=True)
    return df

def _shuffle_dataframe(df):
    index = df.index
    df = shuffle(df)
    df.index = index
    return df

def _split_label_features(df):
    label = df.pop('fraud_ind').to_numpy()
    return df.to_numpy(), label

def split_train_valid(df, percent=0.7):
    fraud = get_fraud(df)
    fraud_size = fraud.shape[0]
    real = get_real(df)
    real_size = real.shape[0]
    fraud1, fraud2 = fraud[:int(fraud_size*percent)], fraud[int(fraud_size*percent):]
    real1, real2 = real[:int(real_size*percent)], real[int(real_size*percent):]
    train = pd.concat([fraud1, real1])
    validation = pd.concat([fraud2, real2])
    
    train = _shuffle_dataframe(train)
    validation = _shuffle_dataframe(validation)

    write_df_to_csv(train, './data/tr.csv')
    write_df_to_csv(validation, './data/va.csv')

    # trX, trY = _split_label_features(train)
    # vaX, vaY = _split_label_features(validation)

    # return trX, trY, vaX, vaY 

def feature_extraction():
    pass

if __name__ == '__main__':
    print('data preprocess')
    '''label status
    target -> fraud_ind
    id -> bacno, txkey, cano
    time -> locdt, loctm
    multi class(<100 class?) -> contp, etymd, stscd, hcefg, csmcu
    multi class (>100 class?) -> (stocn, scity), (mchnom, mcc), acquic
    '''
    df = read_csv_to_df('./data/train_small.csv')
    print(time_interval(df, [('loctm', 6380)]) )
    exit()
    df = preprocess_data(df)
    write_df_to_csv(df, './data/train_small_test.csv')
    trX, trY, vaX, vaY = split_train_valid(df)
    print(trX, trY, vaX, vaY)