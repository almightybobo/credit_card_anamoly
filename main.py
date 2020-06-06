# main script
import data_preprocess
import model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default='./data/tr_nh.csv?format=csv&label_column=4', type=str)
    parser.add_argument('-v', '--valid', default='./data/va_nh.csv?format=csv&label_column=4', type=str)
    parser.add_argument('-m', '--model', default='./model/xgb', type=str)
    parser.add_argument('--model_dir', default='./model', type=str)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--early_stopping_rounds', default=10, type=int)
    parser.add_argument('--subsample', default=0.3, type=float)
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
    clf = model.xgb_model(
            tr,
            va,
            model_path=args.model,
            subsample=args.subsample,
            model_dir=args.model_dir,
            epoch=args.epoch,
            save_period=args.save_period,
            early_stopping_rounds=args.early_stopping_rounds)
