import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from model import tune_hyperparameters, evaluate_model


# 1) SETUP -----------------------------------------------------------------
import sys, os

# Add the parent directory (project root) to Python’s module search path
sys.path.insert(0, os.path.abspath('..'))


# 1️⃣ Load data
df = pd.read_csv("data/processed/features_with_labels.csv")
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

# 2️⃣ Split
from model import split_data
X_train, X_test, y_train, y_test = split_data(X, y)

# 3️⃣ Define both models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
    "RandomForest":      (RandomForestClassifier(),            {"n_estimators": [50,100], "max_depth": [None,10,20]})
}

# 4️⃣ MLflow setup
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("FraudDetection")
client = MlflowClient()

# 5️⃣ Tune and evaluate each, keep track of best RF
best_rf_score = -float("inf")
best_rf_model = None
best_rf_params = None

for name, (model, params) in models.items():
    print(f"→ Training {name}")
    with mlflow.start_run(run_name=name):
        # tune
        tuned_model, tuned_params = tune_hyperparameters(model, params, X_train, y_train)
        # eval
        metrics = evaluate_model(tuned_model, X_test, y_test)
        # log everything
        mlflow.log_params(tuned_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=tuned_model,
            artifact_path="models_tmp",               # temp artifact
            registered_model_name=None                # we’ll register only RF later
        )
        print(f"   {name} metrics:", metrics)
    # if RF, keep best
    if name == "RandomForest" and metrics["accuracy"] > best_rf_score:
        best_rf_score = metrics["accuracy"]
        best_rf_model = tuned_model
        best_rf_params = tuned_params

# 6️⃣ Now register & promote *only* the best RandomForest
if best_rf_model is None:
    raise RuntimeError("RandomForest never ran successfully")

with mlflow.start_run(run_name="RandomForest_Final"):
    # log & register
    mlflow.log_params(best_rf_params)
    mlflow.log_metric("accuracy", best_rf_score)
    res = mlflow.sklearn.log_model(
        sk_model=best_rf_model,
        artifact_path="models",
        registered_model_name="credit_fraud_model"
    )
    print(f"Registered RF as credit_fraud_model, run_id={res.run_id}")

# promote the newly created version
latest = client.get_latest_versions("credit_fraud_model", stages=["None"])
if not latest:
    raise RuntimeError("No un-staged RF model found")
version = latest[0].version
client.transition_model_version_stage(
    name="credit_fraud_model",
    version=version,
    stage="Production",
    archive_existing_versions=True
)
print(f"✅ Promoted credit_fraud_model version {version} → Production")
