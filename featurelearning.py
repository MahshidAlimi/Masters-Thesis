
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mtl.main import *
from mtl.utils import compute_min_W,norm_21
import sklearn.svm as svm
import os
def normalize(data):
    return (data - data.mean())/ data.std()

_INPUT_DIR = os.getcwd() + "\\"
# prevent annoying pandas "A value is trying to be set on a copy of a slice" warnings
pd.options.mode.chained_assignment = None


def Get_Type_Errors(truth, prediction):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(truth)):
        if truth[i] == 1 and prediction[i] == 1:
            TP += 1
        elif truth[i] == 0 and prediction[i] == 1:
            FP += 1
        elif truth[i] == 1 and prediction[i] == 0:
            FN += 1
        else:
            TN += 1

    result = [TP,FP,FN,TN]
    return result

train = False
#Data preparation
df_rating = pd.read_pickle( 'market_yearly.pkl').reset_index()
# df_rating.drop(['Tickerrm', 'Countryrm',
#        'Industryrm', 'Sub_Industryrm', 'Sales_Returnrm', 'AvRatingrm',
#        'ImpliedRatingrm'],axis =1,inplace = True)
df_rating = df_rating[pd.notnull(df_rating["ImpliedRating"])]
df_rating = df_rating[pd.notnull(df_rating["AvRating"])]
cols_drop = ["Country","Sub_Industry"]
_rating_dict = {'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 6}
df_rating["AvRating"] = df_rating["AvRating"].map(_rating_dict)
df_rating["ImpliedRating"] = df_rating["ImpliedRating"].map(_rating_dict)
df_rating.drop(['Ticker','date'],axis = 1,inplace = True)
#df_rating.drop(cols_drop,axis = 1,inplace = True)
for col in cols_drop:
    dummies = pd.get_dummies(df_rating[col])
    df_rating.drop(col,axis=1,inplace=True)
    df_rating[dummies.columns] = dummies


df_rating.Industry = LabelEncoder().fit_transform(df_rating.Industry)
df_rating.rename(columns={'Industry':'tasks'},inplace = True)


#Normalize data/ Fill missing data.
df_rating = df_rating.replace([np.inf, -np.inf], np.nan)
df_rating = df_rating.fillna(df_rating.mean())
df_norm = normalize(df_rating.drop(["ImpliedRating","AvRating"],axis = 1))
df_norm["market_rating_implied"] = df_rating["ImpliedRating"]
df_norm["market_rating_average"] = df_rating["AvRating"]


#Train all models.
classes = df_norm['market_rating_average'].unique()
type_errors = {}
data = df_norm.drop(['market_rating_average', 'market_rating_implied'], axis=1)


tasks= df_norm.tasks.unique()


learning = True
tol = 1e-5



best_score = 0
best_gamma = 0
best_epsilon = 0
best_num = 0
for g in range(-4,3):
    for e in range(-4,3):
        W = np.random.randn(data.shape[1] - 1, len(tasks)) / 100
        W_prev = W
        tol_test = tol + 1
        D = np.eye(data.shape[1] - 1) / (data.shape[1] - 1)
        i = 0
        gamma = 10 ** g
        epsilon = 10 ** e
        while(tol_test > tol and i < 30):
            for t in range(len(tasks)):
                indexes = np.where(data.tasks == tasks[t])
                X_t = data.iloc[indexes].drop('tasks', axis = 1)
                y_t= df_norm["market_rating_average"].iloc[indexes]

                #Compute w for t
                w= compute_min_W(X_t,y_t,D,gamma)
                W[:, t] = w.reshape(-1)

            if learning:
                temp = np.dot(W,W.T) + epsilon * np.eye(len(W))
                U,S,_ = np.linalg.svd(temp)
                temp = np.dot(U,np.dot(np.diag(np.sqrt(S)),U.T))
                D = temp / np.trace(temp)

            else:
                tmp_numer = np.array([np.linalg.norm(W[i,:]) for i in range(D.shape[0])])
                tmp_denom = norm_21(W)
                diag_elem = tmp_numer/tmp_denom
                D = np.diag(diag_elem)
            tol_test = np.linalg.norm(W - W_prev)
            W_prev = W.copy()
            i+=1
        ind_useless = np.where((np.abs(W) < 1e-2).all(axis=1))
        ind_useful = np.setdiff1d(np.arange(W.shape[0]), ind_useless)
        print(len(ind_useful))
        useful = data.columns[ind_useful]
        useless = data.columns[ind_useless]

        X_use = data[useful]
        X_use['tasks'] = data['tasks']
        Xtrain, Xtest, ytrain, ytest = train_test_split(X_use, df_norm["market_rating_average"], test_size=0.2)
        model = svm.SVC()
        model.fit(Xtrain, ytrain)
        accuracy = model.score(Xtest,ytest)
        if accuracy > best_score:
            best_score = accuracy
            best_epsilon = e
            best_gamma = g
            final_features = ind_useful


print(best_score)
print(best_epsilon)
print(best_gamma)
print(len(final_features))

final_X = data[data.columns[ind_useful]]
final_X['Industry'] = data['tasks']
y = df_norm["market_rating_average"]
type_errors = {}
for c in classes:
    for a in classes:
        if c != a and (c,a) not in type_errors:
            print("Getting stats for {} vs {}".format(c,a))
            Y = np.where((y == c) | (y == a))[0]
            y_sub = y.iloc[Y]
            y_sub = np.where(y_sub == c, 1, 0)
            sub_data = final_X.iloc[Y]
            mask = np.random.rand(len(sub_data)) < 0.8
            subx_train = sub_data[mask].values
            subx_test = sub_data[~mask].values

            suby_train = np.asanyarray((y_sub[mask])).reshape(-1, 1)
            suby_test = (y_sub[~mask]).reshape(-1, 1)

            print("Training")
            model = svm.SVC()
            model.fit(subx_train, suby_train)
            prediction = model.predict(subx_test)
            score = model.score(subx_test,suby_test)

            print(score)
            errors = [1 - score]
            t_errors = Get_Type_Errors(suby_test,prediction)
            errors.extend(t_errors)
            type_errors[(c, a)] = errors

import pickle
with open('MTL_ave_Errors.pkl', 'wb') as handle:
        pickle.dump(type_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)