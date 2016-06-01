import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.learning_curve import learning_curve
from sklearn import metrics


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def FactorizeCategoricalVariable(inputDB, categoricalVarName):
    opponentCategories = inputDB[categoricalVarName].value_counts().index.tolist()

    outputDB = pd.DataFrame()
    for category in opponentCategories:
        featureName = categoricalVarName + ': ' + str(category)
        outputDB[featureName] = (inputDB[categoricalVarName] == category).astype(int)

    return outputDB


# load training data

allData = pd.read_csv('./data.csv')
data = allData[allData['shot_made_flag'].notnull()].reset_index()


# add some temporal columns to the data

data['game_date_DT'] = pd.to_datetime(data['game_date'])
data['year'] = data['game_date_DT'].dt.year
data['month'] = data['game_date_DT'].dt.month
data['day'] = data['game_date_DT'].dt.day

data['secondsFromPeriodEnd'] = 60 * data['minutes_remaining'] + data['seconds_remaining']


# select features

featuresDB = pd.DataFrame()
featuresDB['homeGame'] = data['matchup'].apply(lambda x: 1 if (x.find('@') < 0) else 0)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'opponent')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'action_type')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_type')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'combined_shot_type')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_zone_basic')], axis=1)
featuresDB = pd.concat([featuresDB, FactorizeCategoricalVariable(data, 'shot_zone_area')], axis=1)

minmax = preprocessing.MinMaxScaler()
scale = preprocessing.StandardScaler()

featuresDB['period'] = minmax.fit_transform(data['period'].reshape(len(data['period']), 1))
featuresDB['playoffGame'] = minmax.fit_transform(data['playoffs'].reshape(len(data['playoffs']), 1))
featuresDB['locX'] = minmax.fit_transform(data['loc_x'].reshape(len(data['loc_x']), 1))
featuresDB['locY'] = minmax.fit_transform(data['loc_y'].reshape(len(data['loc_y']), 1))
featuresDB['distanceFromBasket'] = minmax.fit_transform(data['shot_distance'].reshape(len(data['shot_distance']), 1))
featuresDB['secondsFromPeriodEnd'] = minmax.fit_transform(
    data['secondsFromPeriodEnd'].reshape(len(data['secondsFromPeriodEnd']), 1))
featuresDB['year'] = minmax.fit_transform(scale.fit_transform(data['year'].reshape(len(data['year']), 1)))
featuresDB['month'] = minmax.fit_transform(scale.fit_transform(data['month'].reshape(len(data['month']), 1)))

labelsDB = data['shot_made_flag']


# prepare models

numFolds = 10

svmLearner = svm.LinearSVC()

nbLearner = naive_bayes.GaussianNB()

lgLearner = linear_model.LogisticRegression()

k_fold = cross_validation.KFold(n=len(labelsDB), n_folds=numFolds)


# start testing

print("--------------Testing Naive Bayes--------------")


predicted = cross_validation.cross_val_predict(nbLearner, featuresDB, labelsDB, cv=k_fold)
print("Accuracy:  " + str(metrics.accuracy_score(labelsDB, predicted)))
print(metrics.classification_report(labelsDB, predicted))

plot_learning_curve(nbLearner, "Naive Bayes Learning Curve", featuresDB, labelsDB, cv=k_fold, n_jobs=4)


print("--------------Testing Logistic Regression--------------")


predicted = cross_validation.cross_val_predict(lgLearner, featuresDB, labelsDB, cv=k_fold)
print("Accuracy:  " + str(metrics.accuracy_score(labelsDB, predicted)))
print(metrics.classification_report(labelsDB, predicted))

plot_learning_curve(lgLearner, "Logistic Regression Learning Curve", featuresDB, labelsDB, cv=k_fold, n_jobs=4)


print("--------------Testing SVM--------------")


predicted = cross_validation.cross_val_predict(svmLearner, featuresDB, labelsDB, cv=k_fold)
print("Accuracy:  " + str(metrics.accuracy_score(labelsDB, predicted)))
print(metrics.classification_report(labelsDB, predicted))

plot_learning_curve(svmLearner, "SVM Learning Curve", featuresDB, labelsDB, cv=k_fold, n_jobs=4)

plt.show()

