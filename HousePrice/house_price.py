import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings('ignore')

train = pd.read_csv('HousePriceData/train.csv')

print(train.info())
print(train.select_dtypes(include=object).columns)
# 1. 分析“SalePrice”
print(train['SalePrice'].median())                      # 回归问题缺失值用中位数替代
sns.distplot(train['SalePrice'])
print('Skewness:{}'.format(train['SalePrice'].skew()))  # 偏度：偏度（Skewness）是描述某变量取值分布对称性的统计量
print('Kurtosis:{}'.format(train['SalePrice'].kurt()))  # 峰度：峰度（Kurtosis）是描述某变量所有取值分布形态陡缓程度的统计量

# 2. 分析特征数据


# 3. 关系矩阵
#
# corrmat = train.corr()
# f, ax = plt.subplots(figsize=(30, 30))
# sns.heatmap(corrmat,vmax=0.6,vmin=0.4,square=True,cmap="RdYlBu_r")

f_names = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']

for x in f_names:
    train[x].fillna('None',inplace=True)
    label = preprocessing.LabelEncoder()
    train[x] = label.fit_transform(train[x])
corrmat1 = train.corr()
f, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(corrmat1, vmax=0.8, vmin=0.2,center=0.5,square=True,cmap="RdYlBu_r")

# k = 11  # 关系矩阵中将显示10个特征
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True,
#                  square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

cols = [
    'OverallQual',
    'YearBuilt',
    'MasVnrArea',
    'TotalBsmtSF',
    'GrLivArea',
    'FullBath',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageYrBlt',
    'GarageCars'
]

print(train[cols].isnull().sum())
for x in ['MasVnrArea','GarageYrBlt']:
    train[x].fillna(train[x].median(),inplace=True)
x = train[cols].values
y = train['SalePrice'].values
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# 之前训练的模型
rfr = clf

test = pd.read_csv('HousePriceData/test.csv')
print(test[cols].isnull().sum())

cars = test['GarageCars'].fillna(test['GarageCars'].median(),inplace=True)
bsmt = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(),inplace=True)
MA = test['MasVnrArea'].fillna(test['MasVnrArea'].median(),inplace=True)
gyb = test['GarageYrBlt'].fillna(test['GarageYrBlt'].median(),inplace=True)
data_test_x = pd.concat([test[cols], cars, bsmt,MA,gyb],axis=1)
print(data_test_x.isnull().sum())

test_x = data_test_x.values
y_te_pred = rfr.predict(test_x)

prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])
result = pd.concat([test['Id'], prediction], axis=1)

plt.show()

result.to_csv('./Predictions.csv', index=False)




