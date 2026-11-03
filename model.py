
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. Load the datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Training rows:", len(train))
print("Testing rows:", len(test))

# 2. Prepare the features
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
X_test = X_test.reindex(columns=X.columns, fill_value=0)
y = train["Cluster"]

# 3. Compare different machine-learning models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = -1
best_model = None
best_name = ""

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    mean_score = scores.mean()
    print(f"{name}: {mean_score:.2%} accuracy")

    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name

# 4. Train the best model using all training data
best_model.fit(X, y)
predictions = best_model.predict(X_test)

# 5. Create Kaggle submission
submission = pd.DataFrame({
    "CustomerID": test["CustomerID"],
    "Cluster": predictions
})

submission.to_csv("submission.csv", index=False)

print("\nBest model:", best_name)
print("Cross-validation accuracy:", f"{best_score:.2%}")
print("Created submission.csv successfully!")
