import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Start tracking an experiment
with mlflow.start_run():
    # Train a model
    model = RandomForestClassifier(max_depth=3)
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log a parameter (e.g., mlets go
    # ax_depth)
    mlflow.log_param("max_depth", 3)

    # Log a metric (e.g., accuracy)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print("Run saved!")

