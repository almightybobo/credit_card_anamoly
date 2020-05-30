## run model and predict
from xgboost.sklearn import XGBClassifier
import pickle

def model(trX, trY, vaX, vaY, model_path='./model/xgb.pkl'):
    clf = XGBClassifier()
    eval_set = [(trX, trY), (valX, valY)]
    clf.fit(trX, trY, 
            early_stopping_rounds=10, 
            eval_set=eval_set, 
            eval_metric='auc', 
            scale_pos_weight=10,
            verbose=True)
    evals_result = clf.evals_result()

def predict(model_path, features):
    loaded_model = load_model(model_path)
    y_pred = loaded_model.predict(features)

def load_model(model_path):
    loaded_model = pickle.load(model_path, 'rb')) 
    return loaded_model

def save_model(model_path):
    if os.path.exists('./model'):
        os.makedirs('./model')
    pickle.dump(clf, open(model_path, 'wb'))
    print('Model saved!')

if __name__ == '__main__':
    print('run model')