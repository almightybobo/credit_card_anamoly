# main script
import data_preprocess
import model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='./data/train.csv', type=str)
    parser.add_argument('-m', '--model', default='', type=str)
    args = parser.parse_args()

    # print('read data')
    # df = data_preprocess.read_csv_to_df(args.data)
    # print('data preprocess')
    # df = data_preprocess.preprocess_data2(df)
    # print('split dataset')
    # data_preprocess.split_train_valid(df)
    # exit()
    # print('get dmatrix')
    tr, va = data_preprocess.get_dmatrix(
            './data/tr_nh.csv?format=csv&label_column=10',
            './data/va_nh.csv?format=csv&label_column=10')
    print('train')
    clf = model.xgb_model(tr, va, model_path=args.model)
