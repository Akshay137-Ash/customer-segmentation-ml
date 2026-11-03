
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

y = train["Cluster"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Extra Trees 200": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "Extra Trees 400": ExtraTreesClassifier(n_estimators=400, random_state=42),
    "Extra Trees Leaf 2": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, random_state=42),
    "Extra Trees Depth 10": ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Random Forest 300": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN 3": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    "KNN 5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
}

best_accuracy = 0
best_model = None
best_name = ""

print("Testing models...\n")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    accuracy = scores.mean() * 100

    print(name, ":", round(accuracy, 2), "%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print("\nBest model:", best_name)
print("Best cross-validation accuracy:", round(best_accuracy, 2), "%")

best_model.fit(X, y)
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    "CustomerID": test["CustomerID"],
    "Cluster": predictions
})

submission.to_csv("submission_improved.csv", index=False)

print("\nCreated submission_improved.csv!")
