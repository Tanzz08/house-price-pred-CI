import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import mlflow
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # file dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))

    input_example = X_train.head(5)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    with mlflow.start_run():
        mlflow.autolog()
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        mlflow.log_params(model.get_params())
        mlflow.log_metric("evaluate_MAE", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("evaluate_MSE", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("evaluate_RMSE", root_mean_squared_error(y_test, y_pred))
        mlflow.log_metric("evaluate_R2", r2_score(y_test, y_pred))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        