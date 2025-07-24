

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Set the tracking URI to your Docker MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # or http://0.0.0.0:5000

# Create experiment if not exists
mlflow.set_experiment("Test Experiment")

# Start a run and log stuff
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")

print("✅ Model logged successfully!")
