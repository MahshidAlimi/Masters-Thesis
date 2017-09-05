
import pandas as pd
import numpy as np
import os
import pickle

import GPy
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

def normalize(data):
    return (data - data.mean())/ data.std()

_INPUT_DIR = os.getcwd() + "\\"
# prevent annoying pandas "A value is trying to be set on a copy of a slice" warnings
pd.options.mode.chained_assignment = None

train = False
#Data preparation
df_rating = pd.read_pickle('market_yearly.pkl')

df_rating = df_rating[pd.notnull(df_rating["ImpliedRating"])]
df_rating = df_rating[pd.notnull(df_rating["AvRating"])]
cols_drop = ["Country","Industry","Sub_Industry"]
_rating_dict = {'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 6}
df_rating["AvRating"] = df_rating["AvRating"].map(_rating_dict)
df_rating["ImpliedRating"] = df_rating["ImpliedRating"].map(_rating_dict)
df_rating.drop(['Ticker','date'],axis = 1,inplace = True)
#df_rating.drop(cols_drop,axis = 1,inplace = True)

for col in cols_drop:
    # dummies = pd.get_dummies(df_rating[col])
    # df_rating.drop(col,axis=1,inplace=True)
    # df_rating[dummies.columns] = dummies
    df_rating[col]=LabelEncoder().fit_transform(df_rating[col])


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
models = {}
# mask = np.random.rand(len(data)) < 0.80
x_train = data
y_train = df_norm['market_rating_average']

# x_test = data[~mask].values
# y_test = df_norm['market_rating_average'][~mask].values.reshape(-1,1)

for c in classes:
    models[c] = []
    for a in classes:
        if c != a and (c,a) not in type_errors:
            print("Getting stats for {} vs {}".format(c,a))
            Y = np.where((y_train == c) | (y_train == a))[0]
            y_sub = y_train.iloc[Y]
            y_sub = np.where(y_sub == c, 1, 0)
            sub_data = x_train.iloc[Y]
            mask = np.random.rand(len(sub_data)) < 0.75
            subx_train = sub_data[mask].values
            subx_test = sub_data[~mask].values

            suby_train = np.asanyarray((y_sub[mask])).reshape(-1, 1)
            suby_test = (y_sub[~mask]).reshape(-1, 1)

            if train:
                subx_train = sub_data.values
                suby_train = y_sub.reshape(-1,1)

            print("Training")
            kernel = GPy.kern.lin(subx_train.shape[1])+GPy.kern.Matern52(subx_train.shape[1])
            GPC = GPy.models.GPClassification(subx_train,suby_train, kernel=kernel)
            GPC.optimize(messages=1)
            if not train:
                print("Test")
                probs = GPC.predict(subx_test)[0]
                prediction = GPy.util.classification.conf_matrix(probs, suby_test)
                print(prediction)
                type_errors[(c,a)] = prediction
            models[c].append(GPC)



#Test the final model
print("Testing for all classes")

mask = np.random.rand(len(data)) < 0.80


x_test = data[~mask].values
y_test = df_norm['market_rating_average'][~mask].values.reshape(-1,1)


label_preds = {}
for label in list(models.keys()):
    preds = None
    for model in models[label]:
        probs = model.predict(x_test)[0]
        if preds is None:
            preds = probs
        else:
            preds += probs

    label_preds[label] = preds

final_predictions = []

for i in range(len(y_test)):
    highest_confidence = -1
    predicted_label = -1
    for label in list(label_preds.keys()):
        prediction = label_preds[label][i]
        if prediction > highest_confidence:
            highest_confidence = prediction
            predicted_label = label

    final_predictions.append(predicted_label)

final_predictions = np.array(final_predictions).reshape(-1,1)
comparison = np.equal(y_test,final_predictions)
final_accuracy = np.sum(comparison)/ len(y_test)
print("Final model accuracy was {} %".format(final_accuracy * 100))


if not train:
    with open('Type_Errors_avm.pkl', 'wb') as handle:
        pickle.dump(type_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
