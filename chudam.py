import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import numpy as np
import os

def train_and_save_model(dataset_path, target_column, pkl_filename):
    # Check if model already exists
    if os.path.exists(pkl_filename):
        print(f"Model already trained. Loading '{pkl_filename}'...")
        model = load(pkl_filename)
        print("Model loaded successfully.")
    else:
        # Load and prepare dataset
        data = pd.read_csv(dataset_path)
        y = data[target_column]
        X = data.drop(target_column, axis=1)

        X.fillna(X.mean(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate and save
        y_pred = model.predict(X_test)
        print(f'Training completion: {model.score(X_test, y_test):.3f}')

        # Save model and scaler
        dump((model, scaler, list(X.columns)), pkl_filename)
        print(f"Model saved as '{pkl_filename}'")

    # Define the predict function (uses loaded or trained model)
    def predict():
        model, scaler, columns = load(pkl_filename)
        X_new = []
        for col in columns:
            val = input(f'Enter value for column {col}: ')
            X_new.append(val)
        X_new = scaler.transform([X_new])
        pred = model.predict(X_new)[0]
        print(f'Predicted outcome: {pred}')

    # Return predict function so user can call it later
    return predict


# Example usage:
predict_func = train_and_save_model("diabetes.csv", "Outcome", "Trained_Logistic_Regression.pkl")
predict_func()  # this will ask for input values and make a prediction




classification_models = [
    # Linear Models
    "LogisticRegression",
    "RidgeClassifier",
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",

    # Naive Bayes
    "GaussianNB",
    "MultinomialNB",
    "ComplementNB",
    "BernoulliNB",
    "CategoricalNB",

    # Neighbors
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",

    # Support Vector Machines
    "SVC",
    "LinearSVC",
    "NuSVC",

    # Trees & Ensembles
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "BaggingClassifier",
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",

    # Discriminant Analysis
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",

    # Neural Networks
    "MLPClassifier",

    # Gaussian Processes
    "GaussianProcessClassifier",

    # Probabilistic / Meta
    "CalibratedClassifierCV",
    "LabelPropagation",
    "LabelSpreading",
    "StackingClassifier",
    "VotingClassifier"
]

regression_models = [
    # Linear Models
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "BayesianRidge",
    "ARDRegression",
    "HuberRegressor",
    "TheilSenRegressor",
    "RANSACRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
    "QuantileRegressor",
    "PassiveAggressiveRegressor",
    "SGDRegressor",

    # Neighbors
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",

    # Support Vector Machines
    "SVR",
    "NuSVR",
    "LinearSVR",

    # Trees & Ensembles
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "BaggingRegressor",
    "AdaBoostRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",

    # Gaussian Processes
    "GaussianProcessRegressor",

    # Neural Networks
    "MLPRegressor",

    # Isotonic Regression
    "IsotonicRegression",

    # Meta / Ensemble Wrappers
    "StackingRegressor",
    "VotingRegressor",
    "TransformedTargetRegressor"
]

clustering_models = [
    "KMeans",
    "MiniBatchKMeans",
    "AgglomerativeClustering",
    "FeatureAgglomeration",
    "DBSCAN",
    "OPTICS",
    "Birch",
    "GaussianMixture",
    "BayesianGaussianMixture",
    "SpectralClustering"
]
