from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, \
    AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)
# pd.set_option("display.height", 1000)
# pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 200)

from sklearn.exceptions import DataConversionWarning

filterwarnings(action="ignore", category=DataConversionWarning)


def main():
    train = pd.read_csv("train.csv", sep=",")

    test = pd.read_csv("test.csv", sep=",")
    # print(train.head())

    train_ids = train["PassengerId"]
    test_ids = test["PassengerId"]

    train = train.drop("PassengerId", axis=1)
    test = test.drop("PassengerId", axis=1)
    # print(train.shape)
    # print(test.shape)

    num_train = train.shape[0]
    num_test = test.shape[0]
    y_train = train.Survived.values

    all_data = pd.concat((train, test), sort=True).reset_index(drop=True)
    # print(all_data.head())
    all_data = all_data.drop(["Survived"], axis=1)

    all_data_missing = all_data.isnull().sum() / len(all_data)
    all_data_missing = all_data_missing.sort_values(ascending=False)
    # print(all_data_missing)
    correlation_matrix = train.corr()
    plt.subplots()
    sns.heatmap(correlation_matrix)
    # plt.show()

    # all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"] \
    #     .transform(lambda x: x.fillna(x.median()))

    columns_to_fill_with_most_frequent = ["Embarked"]
    for column in columns_to_fill_with_most_frequent:
        all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

    columns_to_fill_with_median = ["Age", "Fare"]
    for column in columns_to_fill_with_median:
        all_data[column] = all_data[column].fillna(all_data[column].mean())

    columns_to_drop = ["Cabin", "Name", "Ticket"]
    all_data = all_data.drop(columns_to_drop, axis=1)

    all_data_missing = all_data.isnull().sum() / len(all_data)
    all_data_missing = all_data_missing.sort_values(ascending=False)
    # print(all_data_missing)

    # print(all_data.head())
    columns_to_change_to_categorial = ["Embarked", "Sex"]
    for column in columns_to_change_to_categorial:
        all_data[column] = all_data[column].apply(str)

    all_data["FamilySize"] = all_data["Parch"] + all_data["SibSp"]

    numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # print(skewed_features)

    skewness = pd.DataFrame({"Skew": skewed_features})
    skewness = skewness[abs(skewness) > 0.75]

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # print(skewed_features)

    all_data = pd.get_dummies(all_data)

    all_data = all_data.drop(["Sex_female"], axis=1)
    # print(all_data.head(20))

    train = all_data[:num_train]
    test = all_data[num_train:]

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100),
        "mlp": MLPClassifier(max_iter=1000),
        "logistic": LogisticRegression(solver="lbfgs"),
        "sgd": SGDClassifier(max_iter=1000, tol=1e-3),
        "knn": KNeighborsClassifier(3),
        "svm linear": SVC(kernel="linear", C=0.025),
        "svm rbf": SVC(gamma=2, C=1),
        "gaussian_process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "decision_tree": DecisionTreeClassifier(max_depth=5),
        "ada_boost": AdaBoostClassifier(),
        "naive_bayes": GaussianNB(),
        "qda": QuadraticDiscriminantAnalysis()
    }
    for model_name in models:
        print(model_name)
        model = models[model_name]
        score = cross_val_score(model, train.values, y_train, cv=4, n_jobs=-1)
        print(score.mean(), score.std())

    final_classifier = models["mlp"]
    final_classifier.fit(train.values, y_train)
    predictions = final_classifier.predict(test.values)

    final = pd.DataFrame()
    final["PassengerId"] = test_ids
    final["Survived"] = predictions
    final.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
