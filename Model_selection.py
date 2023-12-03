'''
### Outline

#### how to find input features?
- Use optimize function to see the results of different combinations between different features then choose the top 10 which give ues the best result.
- The input feature we choose is :
        - SMA
        - EMA
        - RSI
        - MACD
        - Bollinger Band
        - Standard Deviation
        - Average Directional Index
        - Rate of Change
        - Momentum
        - ....

#### how to find output target?
- Our target is to find the peak points on which we will execute our strategy to buy and other points are to hold. Using peak function to find the points.

#### how to do model selection?
- first do train-test split, owing that the entire market has the upward trend, it is not appropraite if we do same Standard Scaler in the train and test set, so we split the data to train and test set first then scale data to retain the trend of the price.
- We use 11 algorithms to build our model and do comparason between these models and choose top 3. And we use in-sample and out-of-sample tests to evaluate our models, which contain `errors in training and test set`,  `MSE`,  `RMSE`,  `R_square` and  `AUC` to evaluate the performances of different model.
- Then the best three results are:
        - Logistic Regression
        - Lasso
        - K-Nearest Neighbors
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

Features = pd.read_csv("Features.csv",index_col = [0])
input_features = Features.columns.tolist()[1:-1]
output_feature = Features.columns.tolist()[-1]
print("The input feature: ",input_features)
print("The output feature: ",output_feature)
fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(18, 15))
index = 0
axs = axs.flatten()
for k,v in Features.items():
    sns.boxplot(y=k, data=Features, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(18, 15))
index = 0
axs = axs.flatten()
for k,v in Features.items():
   sns.histplot(v, ax=axs[index],bins = 50)
   index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

sns.pairplot(data=Features.drop(columns = "positions"),
             hue="Close", palette="viridis")
plt.show()

validation_size = 0.2

def preprocess_data(data, validation_size):
    train_size = int(len(data) * (1 - validation_size))
    df_train = data[:train_size]
    df_test = data[train_size:len(data)]

    x_train = df_train[input_features]
    y_train = df_train[output_feature]
    x_test = df_test[input_features]
    y_test = df_test[output_feature]

    x_train = pd.DataFrame(StandardScaler().fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(StandardScaler().fit_transform(x_test), index=x_test.index, columns=x_test.columns)

    return x_train, x_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(Features, validation_size)


def evaluate_fitting(x_train, x_test, y_train, y_test,seed = 1212):
    models = []
    models.append(("Logit", LogisticRegression(random_state=seed)))
    models.append(("Lasso", Lasso(random_state=seed)))
    models.append(("Ridge", Ridge(random_state=seed)))
    models.append(("EN", ElasticNet(random_state=seed)))
    models.append(("SVC", SVC()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier(random_state=seed)))
    models.append(("ETC", ExtraTreesClassifier(random_state=seed)))
    models.append(("RFC", RandomForestClassifier(random_state=seed)))
    models.append(("GBC", GradientBoostingClassifier(random_state=seed)))
    models.append(("ABC", AdaBoostClassifier(random_state=seed)))

    names = []
    train_MSEs = []
    train_RMSEs = []
    train_R2s = []
    test_MSEs = []
    test_R2s = []
    test_RMSEs = []
    names_auc = []
    FPRs = []
    TPRs = []
    ROC_AUCs = []

    for name, model in models:
        names.append(name)
        res = model.fit(x_train, y_train)

        y_pred_train = res.predict(x_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_MSEs.append(train_mse)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_RMSEs.append(train_rmse)
        train_r2 = r2_score(y_train, y_pred_train)
        train_R2s.append(train_r2)

        y_pred_test = res.predict(x_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_MSEs.append(test_mse)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_RMSEs.append(test_rmse)
        test_r2 = r2_score(y_test, y_pred_test)
        test_R2s.append(test_r2)

        ## plot AUC curve
        if hasattr(model, "predict_proba"):
            names_auc.append(name)
            preds = model.predict_proba(x_test)[:, 1]
            fpr, tpr, thresolds = metrics.roc_curve(y_test, preds)
            roc_auc = metrics.auc(fpr, tpr)
            FPRs.append(fpr)
            TPRs.append(tpr)
            ROC_AUCs.append(roc_auc)

    df_train_perform = pd.DataFrame({"MSE": train_MSEs,"RMSE": train_RMSEs,
                                     "R_square": train_R2s}, index=names)
    df_test_perfrom = pd.DataFrame({"MSE": train_MSEs,"RMSE": test_RMSEs,
                                    "R_square": test_R2s}, index=names)
    df_perform = pd.concat([df_train_perform, df_test_perfrom], axis=1, keys=["train", "test"])

    vis_train_test_errors(names, train_MSEs, test_MSEs)
    vis_AUC(names_auc, FPRs, TPRs, ROC_AUCs)
    return df_perform,models

def vis_train_test_errors(names,train_results,test_results):
    fig = plt.figure(figsize = [10,6])
    ind = np.arange(len(names))
    width = 0.30
    fig.suptitle("Comparing the Perfomance of Various Algorithms on the Training vs. Testing Data")
    ax = fig.add_subplot(111)
    plt.bar(ind - width/2,train_results,width = width,label = "Errors in Training Set")
    plt.bar(ind + width/2,test_results,width = width,label = "Errors in Testing Set")
    plt.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    plt.ylabel("Mean Squared Error (MSE)")
    plt.show()

def vis_AUC(names,FPRs,TPRs,ROC_AUCs):
    fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(9, 7))
    axs = axs.flatten()
    for i in range(len(names)):
        axs[i].plot(FPRs[i], TPRs[i], 'b', label='AUC = %0.2f' % ROC_AUCs[i])
        axs[i].legend(loc='lower right')
        axs[i].plot([0, 1], [0, 1], 'r--')
        axs[i].set_xlim([0, 1])
        axs[i].set_ylim([0, 1])
        axs[i].set_ylabel('True Positive Rate')
        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_title('Receiver Operating Characteristic - {}'.format(names[i]))
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.5)
    plt.show()

PERFROMS,models = evaluate_fitting(X_train, X_test, y_train, y_test)
print(PERFROMS)
