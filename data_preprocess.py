# data preprocess
import pandas as pd

def read_csv_to_df(filename):
    df = pd.read_csv(filename)
    return df

def write_df_to_csv(df, filename):
    df.to_csv(filename)

def replace_NY_to_01(df, columns):
    for column in columns:
      df[column].replace('N', '0', inplace=True)
      df[column].replace('Y', '1', inplace=True)
    return df

def normalize_regression_feature(df, columns):
    for column in columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def preprocess_data(df):
    # binary class -> N to 0, Y to 1
    df = replace_NY_to_01(df, ['ecfg', 'insfg', 'ovrlt', 'flbmk', 'flg_3dsmk'])
    # continuous value -> normalize value to 0~1
    df = normalize_regression_feature(df, ['conam', 'iterm'])

def split_train_valid():
    pass

def feature_extraction():
    pass

if __name__ == '__main__':
    print('data preprocess')
    '''label status
    target -> fraud_ind
    id -> bacno, txkey, cano
    time -> locdt, loctm
    multi class(<100 class?) -> contp, etymd, (stocn, scity), stscd, hcefg, csmcu
    multi class (>100 class?) -> (mchnom, mcc), acquic
    '''
    df = read_csv_to_df('./data/train.csv')
    write_df_to_csv(df, './data/train.csv')