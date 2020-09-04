import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from skimage.color import rgb2lab
from skimage.color import lab2rgb
from sklearn.preprocessing import StandardScaler
import sys


def main():
    labelled_data = pd.read_csv(sys.argv[1])
    unlabelled_data = pd.read_csv(sys.argv[2])

    # labelled_data = pd.read_csv("monthly-data-labelled.csv")
    # unlabelled_data = pd.read_csv("monthly-data-unlabelled.csv")

    X_labelled = labelled_data.drop(['city', 'year'], axis=1)
    y_labelled = labelled_data["city"]
    X_train, X_valid, y_train, y_valid = train_test_split(X_labelled, y_labelled)

    bayes_model = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )
    bayes_model.fit(X_train, y_train)
    print("Bayes model train", bayes_model.score(X_train, y_train))
    print("Bayes model valid", bayes_model.score(X_valid, y_valid))

    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors = 10)
    )
    knn_model.fit(X_train, y_train)
    print("Knn model train", knn_model.score(X_train, y_train))
    print("Knn model valid", knn_model.score(X_valid, y_valid))

    rf_model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10)
    )
    rf_model.fit(X_train, y_train)
    print("RF model train", rf_model.score(X_train, y_train))
    print("RF model valid", rf_model.score(X_valid, y_valid))


    unlabelled_data = unlabelled_data.drop(['city', 'year'], axis=1)
    prediction = knn_model.predict(unlabelled_data)

    pd.Series(prediction).to_csv("labels.csv", index=False, header=False)
    #pd.Series(answer).to_csv(sys.argv[3], index=False, header=False)


if __name__=="__main__":
    main()
