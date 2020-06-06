# main script
import data_preprocess
import model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default='./data/tr_nh.csv?format=csv&label_column=4', type=str)
    parser.add_argument('-v', '--valid', default='./data/va_nh.csv?format=csv&label_column=4', type=str)
    parser.add_argument('-m', '--model', default='./model/xgb', type=str)
    args = parser.parse_args()

    # print('read data')
    # df = data_preprocess.read_csv_to_df(args.data)
    # print('data preprocess')
    # df = data_preprocess.preprocess_data2(df)
    # print(df.columns.get_loc('fraud_ind'))
    # exit()
    # print('split dataset')
    # data_preprocess.split_train_valid(df)
    # exit()
    print('get dmatrix')
    tr, va = data_preprocess.get_dmatrix(
            args.train,
            args.valid)
    print('train')
    clf = model.xgb_model(tr, va, model_path=args.model)
