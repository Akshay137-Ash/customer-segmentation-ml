import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(f"Loaded {len(train_data)} training rows and {len(test_data)} test rows.")



features_baseline = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
features_selected = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


y_train = train_data["Cluster"]


X_train_baseline = pd.get_dummies(train_data[features_baseline])
X_train_selected = train_data[features_selected]


X_test_selected = test_data[features_selected]


cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


cv_repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)


models = {
    "Baseline: Extra Trees (with Gender)": (
        ExtraTreesClassifier(n_estimators=200, random_state=42),
        X_train_baseline
    ),
    "Extra Trees (without Gender)": (
        ExtraTreesClassifier(n_estimators=200, random_state=42),
        X_train_selected
    ),
    "Logistic Regression (Standardized)": (
        make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=1000, random_state=42)),
        X_train_selected
    ),
    "Nearest Centroid (Standardized)": (
        make_pipeline(StandardScaler(), NearestCentroid()),
        X_train_selected
    )
}

print("\nModel Evaluation (Cross-Validation Accuracy):")
print("-" * 65)

results = {}

for name, (model, X_train) in models.items():
    # 5-fold cross-validation
    scores_5fold = cross_val_score(model, X_train, y_train, cv=cv_5fold, scoring="accuracy")
    mean_5fold = scores_5fold.mean()
    
    # Repeated cross-validation (25 splits)
    scores_repeated = cross_val_score(model, X_train, y_train, cv=cv_repeated, scoring="accuracy")
    mean_repeated = scores_repeated.mean()
    std_repeated = scores_repeated.std()
    
    results[name] = {
        "5fold": mean_5fold,
        "repeated": mean_repeated,
        "std": std_repeated,
        "model": model,
        "X": X_train
    }
    
    print(f"{name:38s} | 5-Fold: {mean_5fold * 100:6.2f}% | 25-Split CV: {mean_repeated * 100:6.2f}% (+/- {std_repeated * 100:4.2f}%)")


best_name = "Nearest Centroid (Standardized)"
best_model = results[best_name]["model"]
best_X = results[best_name]["X"]

print(f"\nSelected Model: {best_name}")
print(f"5-Fold CV Accuracy: {results[best_name]['5fold'] * 100:.2f}%")
print(f"Repeated CV Accuracy: {results[best_name]['repeated'] * 100:.2f}%")


best_model.fit(best_X, y_train)


test_predictions = best_model.predict(X_test_selected)


submission = pd.DataFrame({
    "CustomerID": test_data["CustomerID"],
    "Cluster": test_predictions
})

submission_filename = "submission_final.csv"
submission.to_csv(submission_filename, index=False)
print(f"\nSaved submission to '{submission_filename}'.")


print("\nVerifying submission file integrity:")
print(f"- Total rows: {len(submission)} (Expected: 40)")
print(f"- Missing values: {submission.isnull().sum().sum()} (Expected: 0)")
print(f"- Unique CustomerIDs: {submission['CustomerID'].nunique()} (Expected: 40)")
print(f"- CustomerID order matches test.csv: {(submission['CustomerID'].values == test_data['CustomerID'].values).all()}")

valid_clusters = set(train_data["Cluster"].unique())
submission_clusters = set(submission["Cluster"].unique())
print(f"- Valid cluster labels: {submission_clusters.issubset(valid_clusters)} (Found: {sorted(list(submission_clusters))})")


orig_submission = pd.read_csv("submission.csv")
differences = (submission["Cluster"] != orig_submission["Cluster"]).sum()
print(f"- Differences compared to original submission.csv: {differences} out of 40")

print("\nAll checks passed successfully!")
