import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.drop('QuoteNumber', axis=1)
test = test.drop('QuoteNumber', axis=1)

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: x.year)
train['Month'] = train['Date'].apply(lambda x: x.month)
train['weekday'] = train['Date'].apply(lambda x: x.weekday())

test['Year'] = test['Date'].apply(lambda x: x.year)
test['Month'] = test['Date'].apply(lambda x: x.month)
test['weekday'] = test['Date'].apply(lambda x: x.weekday())

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-999)
test = test.fillna(-999)

features = list(train.columns[1:])

for f in train.columns:
    if train[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

xgb_model = xgb.XGBClassifier()

parameters = {
    'nthread': [4],
    'objective': ['binary:logistic'],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    # 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
    'reg_alpha': [0.001],
    'gamma': [0],
    'max_depth': [15],
    'min_child_weight': [9],  # 4,9,16
    'silent': [1],
    'subsample': [0.83],
    'colsample_bytree': [0.77],
    'n_estimators': [100],
    'missing': [-999],
    'seed': [2000]
}

clf = GridSearchCV(xgb_model, parameters, n_jobs=1,
                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train[features], train["QuoteConversion_Flag"])

print(xgb_model)
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(test[features])[:, 1]

sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)
