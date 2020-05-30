# data visualization
import os
import numpy as np
import matplotlib.pyplot as plt
import data_preprocess

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

def count_each_column(df):
    size = []
    for c in df.columns:
        size.append(df[c].value_counts().size)
        print("{}: {}".format(c, size[-1]))
    
    for s, c in zip(size, df.columns):
        if s > 10:
            continue
        print("---- %s ---" % c)
        print(df[c].value_counts())

def binary_histogram(df, col_names, save=True, save_name='example'):
    N, Y = [], []
    for col_name in col_names:
        print(col_name)
        count = df[col_name].value_counts().to_list()
        N.append(count[0])
        Y.append(count[1]) if len(count) == 2 else Y.append(0)

    width = 0.3
    plt.bar([i-width/2 for i, col_name in enumerate(col_names)], N, width=width, label='N')
    plt.bar([i+width/2 for i, col_name in enumerate(col_names)], Y, width=width, label='Y')
    plt.xticks(range(len(col_names)), col_names)
    if save:
        save_path = os.path.join('pic', save_name)
        plt.savefig(save_path)


if __name__ == '__main__':
    print('data analysis')
    df = data_preprocess.read_csv_to_df('./data/train_small.csv')
    # count_each_column(df)

    # split fraud and real data
    fraud = get_fraud(df)
    binary_histogram(fraud, ['ecfg', 'insfg', 'ovrlt', 'flbmk', 'flg_3dsmk'], True, 'fraud_binary')
    real = get_real(df)
    binary_histogram(real, ['ecfg', 'insfg', 'ovrlt', 'flbmk', 'flg_3dsmk'], True, 'real_binary')

    
    