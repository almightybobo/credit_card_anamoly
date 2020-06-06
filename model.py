## run model and predict
import xgboost as xgb
import pickle
import os

def xgb_model(tr, va, model_path='./model/xgb', **kwargs):
    params = {
            'subsample': kwargs.get('subsample', 0.3), 
            'scale_pos_weight': 20, 
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr'}
    watch_list = [(tr, 'train'), (va, 'eval')]
    clf = xgb.train(params, tr, 100, watch_list, early_stopping_rounds=10)
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