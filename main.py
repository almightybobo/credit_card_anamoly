# main script
import data_preprocess
import model

if __name__ == '__main__':
    df = data_preprocess.read_csv_to_df('./data/train.csv')
    df = data_preprocess.preprocess_data1(df)
    trX, trY, vaX, vaY = data_preprocess.split_train_valid(df)
    clf = model.xgb_model(trX, trY, vaX, vaY, model_path='./model/xgb1')
