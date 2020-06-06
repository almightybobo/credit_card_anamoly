# main script
import data_preprocess
import model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='./data/train.csv', type=str)
    parser.add_argument('-m', '--model', default='', type=str)
    args = parser.parse_args()

    print('read data')
    df = data_preprocess.read_csv_to_df(args.data)
    print('data preprocess')
    df = data_preprocess.preprocess_data2(df)
    print('train')
    trX, trY, vaX, vaY = data_preprocess.split_train_valid(df)
    clf = model.xgb_model(trX, trY, vaX, vaY, model_path=args.model)
