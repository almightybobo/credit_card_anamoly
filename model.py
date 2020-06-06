## run model and predict
import xgboost as xgb
import pickle
import os

def xgb_model(tr, va, model_path, **kwargs):
    params = {
            'subsample': kwargs.get('subsample', 0.3), 
            'scale_pos_weight': 20, 
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr'}
    watch_list = [(tr, 'train'), (va, 'eval')]

    if os.path.exists(model_path):
        clf_name = model_path
    else:
        clf_name = None
    for e in range(kwargs.get('epoch', 1000) // kwargs.get('save_period', 5)):
        clf = xgb.train(
                params, 
                tr, 
                kwargs.get('save_period', 5), 
                watch_list, 
                xgb_model=clf_name,
                early_stopping_rounds=kwargs.get('early_stopping_rounds', 10))
        clf_name = model_path + '_%d' % e
        clf.save_model(clf_name)
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