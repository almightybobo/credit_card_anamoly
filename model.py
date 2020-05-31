## run model and predict
from xgboost.sklearn import XGBClassifier
import pickle

def xgb_model(trX, trY, vaX, vaY, model_path='./model/xgb.pkl'):
    clf = XGBClassifier(n_estimators=50, scale_pos_weight=10)
    eval_set = [(trX, trY), (vaX, vaY)]
    clf.fit(trX, trY, 
            early_stopping_rounds=10, 
            eval_set=eval_set, 
            eval_metric='aucpr', 
            verbose=True)

    save_model(clf, model_path)
    return clf

def predict(model_path, features):
    loaded_model = load_model(model_path)
    y_pred = loaded_model.predict(features)

def load_model(model_path):
    loaded_model = pickle.load(model_path, 'rb')
    return loaded_model

def save_model(clf, model_path):
    if os.path.exists('./model'):
        os.makedirs('./model')
    pickle.dump(clf, open(model_path, 'wb'))
    print('Model saved!')

if __name__ == '__main__':
    print('run model')