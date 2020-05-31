# data preprocess
import sklearn
import pandas as pd

def read_csv_to_df(filename):
    df = pd.read_csv(filename)
    return df

def write_df_to_csv(df, filename):
    df.to_csv(filename, index=False)

def replace_NY_to_01(df, columns):
    for column in columns:
      df[column].replace('N', '0', inplace=True)
      df[column].replace('Y', '1', inplace=True)
    return df

def normalize_regression_feature(df, columns):
    for column in columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def one_hot_encoding(df, columns):
    for column in columns:
        df = pd.concat([df,pd.get_dummies(df[column], prefix=column)],axis=1)
        df.drop([column],axis=1, inplace=True)
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

def preprocess_data(df):
    # binary class -> N to 0, Y to 1
    df = replace_NY_to_01(df, ['ecfg', 'insfg', 'ovrlt', 'flbmk', 'flg_3dsmk'])
    # continuous value -> normalize value to 0~1
    df = normalize_regression_feature(df, ['conam', 'iterm'])
    df = one_hot_encoding(df, ['contp', 'etymd', 'stscd', 'hcefg', 'csmcu'])
    return df

def split_train_valid(df, percent=0.7):
    fraud = get_fraud(df)
    fraud_size = fraud.shape[0]
    real = get_real(df)
    real_size = real.shape[0]
    fraud1, fraud2 = fraud[: int(fraud_size*percent)], fraud[int(fraud_size*percent):]
    real1, real2 = real[: int(real_size*percent)], fraud[int(real_size*percent):]
    train = pd.concat([fraud1, real1])
    validation = pd.concat([fraud2, real2])

    return train, validation

def feature_extraction():
    pass

if __name__ == '__main__':
    print('data preprocess')
    '''label status
    target -> fraud_ind
    id -> bacno, txkey, cano
    time -> locdt, loctm
    multi class(<100 class?) -> contp, etymd, , stscd, hcefg, csmcu
    multi class (>100 class?) -> (stocn, scity), (mchnom, mcc), acquic
    '''
    df = read_csv_to_df('./data/train_small.csv')
    df = preprocess_data(df)
    train, validation = split_train_valid(df)
    print(train, validation)
    write_df_to_csv(df, './data/train_small_test.csv')