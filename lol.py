import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
import time


def decision_tree(X, y):

    # Range used for Grid Search
    # max_depth_range = range(3, 12)

    # Best parameters after grid search
    max_depth_range = [8]

    param_grid = dict(max_depth=max_depth_range)

    # GridsearchCV
    clf = DecisionTreeClassifier()
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_


def random_forest_model(X, y):

    # Range used for Grid Search
    # n_estimators_range = [200]
    # max_features_range = range(3, 15) # range(8, 10)
    # max_depth = range(5, 10)

    # Best parameters after grid search
    n_estimators_range = [200]
    max_features_range = [5]
    max_depth_range = [9]

    param_grid = dict(n_estimators=n_estimators_range, max_features=max_features_range, max_depth=max_depth_range)

    # GridsearchCV
    clf = RandomForestClassifier()
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_


def adaboost_model(X, y):

    # Range used for Grid Search
    # n_estimators_range = range(10, 101, 10)
    # learning_rate_range = [0.7, 0.8, 0.9]

    # Best parameters after grid search
    n_estimators_range = [100]
    learning_rate_range = [0.8]

    param_grid = dict(n_estimators=n_estimators_range, learning_rate=learning_rate_range)

    # GridsearchCV
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2))
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_


def model_train_and_evaluation(X_train, y_train, X_test, y_test, model):
    start = time.time()

    if model == 'dt':
        clf, params = decision_tree(X_train, y_train)
    elif model == 'rf':
        clf, params = random_forest_model(X_train, y_train)
    elif model == 'ada':
        clf, params = adaboost_model(X_train, y_train)
    elif model == 'mlp':

        # standardization before mlp model
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = MLPClassifier(learning_rate_init=0.001, hidden_layer_sizes=(100,)).fit(X_train, y_train)
        params = clf.get_params()

    end = time.time()
    print('Training Time: ', end - start)

    y_pred = clf.predict(X_test)

    print(params)
    print('AUC score: {}'.format(roc_auc_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred, digits=7))


data = pd.read_csv(r"D:\python projects\Big Data project1\new_data.csv")
test_data = pd.read_csv(r"D:\python projects\Big Data project1\test_set.csv")

# data view
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
print(data.head(5))
print(data.describe())
print(data.info())

X_train = data.drop(['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner'], axis=1).values
y_train = data['winner'].values
X_test = test_data.drop(['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner'], axis=1).values
y_test = test_data['winner'].values

cls = ['dt', 'rf', 'ada', 'mlp']
for c in cls:
    model_train_and_evaluation(X_train, y_train, X_test, y_test, c)

