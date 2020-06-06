## run model and predict
from xgboost.sklearn import XGBClassifier
import pickle
import os

def xgb_model(trX, trY, vaX, vaY, model_path='./model/xgb'):
    clf = XGBClassifier(n_estimators=100, scale_pos_weight=10, subsample=0.1)
    eval_set = [(trX, trY), (vaX, vaY)]
    clf.fit(trX, trY, 
            early_stopping_rounds=10, 
            eval_set=eval_set, 
            eval_metric='aucpr', 
            xgb_model=model_path if os.path.exists(model_path) else None,
            verbose=True)

    clf.save_model(model_path)
    return clf

def predict(model_path, features):
    clf = load_model(model_path)
    y_pred = clf.predict(features)
    return y_pred

def load_model(model_path):
    clf = XGBClassifier()
    clf = clf.load_model(model_path)
    return clf

def save_model(clf, model_path):
    if not os.path.exists('./model'):
        os.makedirs('./model')
    pickle.dump(clf, open(model_path, 'wb'))
    print('Model saved!')

if __name__ == '__main__':
    print('run model')