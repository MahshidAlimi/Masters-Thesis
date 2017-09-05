import pandas as pd
import numpy as np
import os
import pickle

from sklearn.svm import SVC
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
#from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.datasets import make_classification as score
import warnings
import sklearn.metrics as sm
from sklearn.datasets import make_classification
warnings.filterwarnings('ignore')


def normalize(data):
    return (data - data.mean())/ data.std()

_INPUT_DIR = os.getcwd() + "\\"
# prevent annoying pandas "A value is trying to be set on a copy of a slice" warnings
pd.options.mode.chained_assignment = None

train = False
df_rating = pd.read_pickle('market_yearly.pkl')
# type_errors = {}
df_rating = df_rating[pd.notnull(df_rating["ImpliedRating"])]
df_rating = df_rating[pd.notnull(df_rating["AvRating"])]
cols = ["Country","Industry","Sub_Industry"]
_rating_dict = {'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 6}
df_rating["AvRating"] = df_rating["AvRating"].map(_rating_dict)
df_rating["ImpliedRating"] = df_rating["ImpliedRating"].map(_rating_dict)
df_rating.drop(['Ticker','date'],axis = 1,inplace = True)

for col in cols:
    dummies = pd.get_dummies(df_rating[col])
    df_rating.drop(col,axis=1,inplace = True)
    df_rating[dummies.columns] = dummies

df_rating = df_rating.replace([np.inf, -np.inf], np.nan)
df_rating = df_rating.fillna(df_rating.mean())
df_norm = normalize(df_rating.drop(["ImpliedRating","AvRating"],axis = 1))
df_norm["market_rating_implied"] = df_rating["ImpliedRating"]
df_norm["market_rating_average"] = df_rating["AvRating"]

data = df_norm.drop(['market_rating_average', 'market_rating_implied'], axis=1)

print("Testing for all classes")
mask = np.random.rand(len(data)) < 0.75
x_train = data[mask].values
x_test = data[~mask].values

y = df_norm["market_rating_average"]
y_train = np.asanyarray((y[mask])).reshape(-1,1)
y_test = (y[~mask]).reshape(-1,1)

clf = SVC(decision_function_shape='ovo',kernel = 'rbf')
clf.fit(x_test,y_test)
print("SVM test score {}".format(clf.score(x_test,y_test)))

y_pred = clf.predict(x_test).reshape([-1,])
print(y_pred [0:5])
print(y_test.reshape([-1,])[0:5])


class_1_pred = (y_pred==0)
class1_true = (y_test ==0)
print(class_1_pred[0:5])
print(class1_true.reshape([-1,])[0:5])
precision1 =sm.precision_score(class1_true,class_1_pred )
recall1 =sm.recall_score(class1_true,class_1_pred )


print('precision: {}'.format(precision1))
print('recall: {}'.format(recall1))


class_2_pred = (y_pred==1)
class2_true = (y_test ==1)
print(class_2_pred[0:5])
print(class2_true.reshape([-1,])[0:5])
precision2 =sm.precision_score(class2_true,class_2_pred )
recall2 =sm.recall_score(class2_true,class_2_pred )


print('precision: {}'.format(precision2))
print('recall: {}'.format(recall2))


class_3_pred = (y_pred==2)
class3_true = (y_test ==2)
print(class_3_pred[0:5])
print(class3_true.reshape([-1,])[0:5])
precision3 =sm.precision_score(class3_true,class_3_pred )
recall3 =sm.recall_score(class3_true,class_3_pred )


print('precision: {}'.format(precision3))
print('recall: {}'.format(recall3))


class_4_pred = (y_pred==3)
class4_true = (y_test ==3)
print(class_4_pred[0:5])
print(class4_true.reshape([-1,])[0:5])
precision4 =sm.precision_score(class4_true,class_4_pred )
recall4 =sm.recall_score(class4_true,class_4_pred )


print('precision: {}'.format(precision4))
print('recall: {}'.format(recall4))


class_5_pred = (y_pred==4)
class5_true = (y_test ==4)
print(class_5_pred[0:5])
print(class5_true.reshape([-1,])[0:5])
precision5 =sm.precision_score(class5_true,class_5_pred )
recall5 =sm.recall_score(class5_true,class_5_pred )


print('precision: {}'.format(precision5))
print('recall: {}'.format(recall5))


class_6_pred = (y_pred==5)
class6_true = (y_test ==5)
print(class_6_pred[0:5])
print(class6_true.reshape([-1,])[0:5])
precision6 =sm.precision_score(class6_true,class_6_pred )
recall6 =sm.recall_score(class6_true,class_6_pred )


print('precision: {}'.format(precision6))
print('recall: {}'.format(recall6))

class_7_pred = (y_pred==6)
class7_true = (y_test ==6)
print(class_7_pred[0:5])
print(class7_true.reshape([-1,])[0:5])
precision7 =sm.precision_score(class7_true,class_7_pred )
recall7 =sm.recall_score(class7_true,class_7_pred )


print('precision: {}'.format(precision7))
print('recall: {}'.format(recall7))


#
# if not train:
#     with open('svm_Errors_av.pkl', 'wb') as handle:
#         pickle.dump(type_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
