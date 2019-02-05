import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)


def main():
    train_X, train_y, test_X, test_y = read_and_prepare_data(clean=False)

    # test_classifiers(train_X, train_y)
    clf = grid_search_over_RFC(train_X, train_y)
    # clf = grid_search_over_KNN(train_X, train_y)
    classify(clf, test_X, test_y)


def classify(clf, test_X, test_y):
    out = clf.predict(test_X)
    print("PassengerId,Survived")
    for idx, o in enumerate(out):
        print(test_y[idx], ",", o, sep="")


def grid_search_over_KNN(train_X, train_y):
    parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], "weights": ["uniform", "distance"]}
    knn = KNeighborsClassifier()

    clf = GridSearchCV(knn, parameters, cv=10)
    clf.fit(train_X, train_y)

    # print(clf.cv_results_['mean_test_score'])
    return clf.best_estimator_


def grid_search_over_RFC(train_X, train_y):
    parameters = {'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]}
    rfc = RandomForestClassifier(class_weight="balanced", bootstrap=True, oob_score=True,
                                 criterion='gini')

    clf = GridSearchCV(rfc, parameters, cv=10)
    clf.fit(train_X, train_y)

    # print(clf.cv_results_['mean_test_score'])
    return clf.best_estimator_


def test_classifiers(train_X, train_y):
    classifiers = [DecisionTreeClassifier(),
                   RandomForestClassifier(n_estimators=300, class_weight="balanced", bootstrap=True, oob_score=True,
                                          criterion='gini'),
                   SVC(kernel='linear', gamma="scale", class_weight="balanced"),
                   SVC(kernel="poly", gamma="scale", class_weight="balanced"),
                   SVC(kernel="rbf", gamma="scale", class_weight="balanced"),
                   SVC(kernel="sigmoid", gamma="scale", class_weight="balanced"),
                   LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced"),
                   SGDClassifier(max_iter=200, tol=1e-3, class_weight="balanced"),
                   MLPClassifier(max_iter=200),
                   KNeighborsClassifier(n_neighbors=7),
                   ]
    for clf in classifiers:
        score = np.mean(cross_val_score(clf, train_X, train_y, cv=10, n_jobs=-1,
                                        scoring=make_scorer(balanced_accuracy_score, adjusted=True)))
        print(score)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def read_and_prepare_data(clean=False):
    df_train = pd.read_csv('train.csv', sep=",")

    df_test = pd.read_csv('test.csv', sep=",")
    # print(df.describe())

    df_train['Family_size'] = df_train['SibSp'] + df_train['Parch'] + 1
    df_test['Family_size'] = df_test['SibSp'] + df_test['Parch'] + 1

    column_names = ["Sex"]
    for col in column_names:
        df_train = hot_encode_column(col, df_train)
        df_test = hot_encode_column(col, df_test)

    column_names = ["Pclass", "Age", "Fare", "Family_size"]
    for col in column_names:
        scaler = StandardScaler()
        scaler.fit(df_train[[col]])
        scale(col, df_train, scaler)
        scale(col, df_test, scaler)

    column_names = ["Name", "Ticket", "Cabin", "Embarked", "Sex_female", "SibSp", "Parch"]
    df_train = df_train.drop(labels=column_names, axis=1)
    df_test = df_test.drop(labels=column_names, axis=1)

    if clean:
        clean_dataset(df_train)
        clean_dataset(df_test)
    else:
        df_train = df_train.fillna(0)
        df_test = df_test.fillna(0)

    train_y_labels = df_train.columns[[1]]
    train_x_labels = df_train.columns[2:]

    test_y_labels = df_test.columns[[0]]
    test_x_labels = df_test.columns[1:]

    X = df_train.filter(train_x_labels)
    y = df_train.filter(train_y_labels)
    train_X = X.values
    train_y = y.values.ravel()

    X = df_test.filter(test_x_labels)
    y = df_test.filter(test_y_labels)
    test_X = X.values
    test_y = y.values.ravel()

    return train_X, train_y, test_X, test_y


def scale(col, df, scaler):
    columns = df[[col]]
    scaled_values = scaler.fit_transform(columns)
    df[col] = scaled_values


def hot_encode_column(column_name, df):
    new_cols = pd.get_dummies(df[column_name], prefix=column_name)
    df = df.drop(column_name, axis=1)
    df = df.join(new_cols)
    return df


if __name__ == "__main__":
    main()
