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
    test_classifiers()
    # clf = grid_search_over_RFC()

    # clf = grid_search_over_KNN()
    # classify(clf)


def classify(clf):
    X, y = read_and_prepare_data(test_or_train="train", clean=False)
    clf.fit(X, y)
    X, y = read_and_prepare_data(test_or_train="test", clean=False)
    out = clf.predict(X)
    print("PassengerId,Survived")
    for idx, o in enumerate(out):
        print(y[idx], ",", o, sep="")


def grid_search_over_KNN():
    X, y = read_and_prepare_data(clean=False)

    parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], "weights": ["uniform", "distance"]}
    knn = KNeighborsClassifier()

    clf = GridSearchCV(knn, parameters, cv=10)
    clf.fit(X, y)

    # print(clf.cv_results_['mean_test_score'])
    return clf.best_estimator_


def grid_search_over_RFC():
    X, y = read_and_prepare_data(clean=False)

    parameters = {'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]}
    rfc = RandomForestClassifier(class_weight="balanced", bootstrap=True, oob_score=True,
                                 criterion='gini')

    clf = GridSearchCV(rfc, parameters, cv=10)
    clf.fit(X, y)

    # print(clf.cv_results_['mean_test_score'])
    return clf.best_estimator_


def test_classifiers():
    X, y = read_and_prepare_data(clean=False)

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
        score = np.mean(cross_val_score(clf, X, y, cv=10, n_jobs=-1,
                                        scoring=make_scorer(balanced_accuracy_score, adjusted=True)))
        print(score)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def read_and_prepare_data(test_or_train="train", clean=True):
    if test_or_train == "train":
        df = pd.read_csv('train.csv', sep=",")
    else:
        df = pd.read_csv('test.csv', sep=",")
    # print(df.describe())

    column_names = ["Sex", "SibSp", "Parch"]
    for col in column_names:
        df = hot_encode_column(col, df)

    column_names = ["Pclass", "Age", "Fare"]
    for col in column_names:
        standard_scale_column(col, df)

    df = df.drop(labels=["Name", "Ticket", "Cabin", "Embarked", "Sex_female"], axis=1)

    if clean:
        clean_dataset(df)
    else:
        df = df.fillna(0)
    # print(df)
    if test_or_train == "train":
        y_labels = df.columns[[1]]
        x_labels = df.columns[2:]
    else:
        y_labels = df.columns[[0]]
        x_labels = df.columns[1:]
    # print(x_labels, y_labels)
    X = df.filter(x_labels)
    y = df.filter(y_labels)
    X = X.values
    y = y.values.ravel()

    return X, y


def standard_scale_column(col, df):
    scaler = StandardScaler()
    scale(col, df, scaler)


def min_max_scale_column(col, df):
    scaler = StandardScaler()
    scale(col, df, scaler)


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
