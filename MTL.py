import pandas as pd
import numpy as np
import os
import pickle
import GPy

from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


#This is the script for the  MTL.

def accuracy(truth,prediction):
    comparison = np.equal(truth,prediction)
    acc = sum(comparison) / truth.shape[0]
    return acc

def normalize(data):
    return (data - data.mean())/ data.std()

_INPUT_DIR = os.getcwd() + "\\"
# prevent annoying pandas "A value is trying to be set on a copy of a slice" warnings
pd.options.mode.chained_assignment = None

train = False
df_rating = pd.read_pickle('market_yearly.pkl')

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

y1 = df_norm["market_rating_implied"]
y1_train = np.asanyarray((y1[mask])).reshape(-1,1)
y1_test = (y1[~mask]).reshape(-1,1)

y2 = df_norm["market_rating_average"]
y2_train = np.asanyarray((y2[mask])).reshape(-1,1)
y2_test = (y2[~mask]).reshape(-1,1)

implied_accuracies =[]
average_accuracies = []
for i in range(1):
    K3 = GPy.kern.Matern32(x_train.shape[1])
    lcm = GPy.util.multioutput.LCM(input_dim=x_train.shape[1],num_outputs=2,kernels_list = [K3])
    m = GPy.models.GPCoregionalizedRegression([x_train, x_train], [y1_train, y2_train], kernel=lcm)
    m.optimize()
    newXtest = np.hstack([x_test,np.ones_like(x_test)])
    noise_dict = {'output_index':newXtest[:,newXtest.shape[1]-1:].astype(int)}
    pred = m.predict(newXtest,Y_metadata=noise_dict)
    implied_accuracies.append(accuracy(y1_test.astype(np.int8),pred[0].astype(np.int8)))
    average_accuracies.append(accuracy(y2_test.astype(np.int8),pred[0].astype(np.int8)))

implied_mean = sum(implied_accuracies)/len(implied_accuracies)
average_mean = sum(average_accuracies)/len(average_accuracies)
print("Mean implied accuracy: {}%".format(implied_mean * 100))
print("Mean average accuracy: {}%".format(average_mean * 100))