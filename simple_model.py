
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("OSPC Customer Segmentation")
print("--------------------------")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("\nTraining data:")
print(train.head())

print("\nTesting data:")
print(test.head())

print("\nNumber of training customers:", len(train))
print("Number of testing customers:", len(test))

print("\nTraining columns:")
print(train.columns.tolist())

print("\nMissing values:")
print(train.isnull().sum())

features = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = train[features]
y = train["Cluster"]

X_test = test[features]

print("\nCluster distribution:")
print(y.value_counts().sort_index())

print("\nConverting Gender into numeric values...")

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("\nFeatures used:")
print(X.columns.tolist())

print("\nCreating the model...")

model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

print("\nChecking model accuracy...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

for i in range(len(scores)):
    print("Fold", i + 1, "Accuracy:", round(scores[i] * 100, 2), "%")

average_accuracy = scores.mean() * 100

print("\nAverage accuracy:", round(average_accuracy, 2), "%")

print("\nTraining model using all training customers...")

model.fit(X, y)

print("Training completed!")

print("\nPredicting customer clusters...")

predictions = model.predict(X_test)

print("Prediction completed!")

print("\nFirst 10 predictions:")
print(predictions[:10])

submission = pd.DataFrame()

submission["CustomerID"] = test["CustomerID"]
submission["Cluster"] = predictions

print("\nChecking submission...")

if len(submission) == len(test):
    print("Correct number of rows!")
else:
    print("Row count problem!")

if submission.isnull().sum().sum() == 0:
    print("No missing values!")
else:
    print("Missing values found!")

submission.to_csv("submission_basic.csv", index=False)

print("\nSubmission preview:")
print(submission.head())

print("\nFile saved: submission_basic.csv")
print("Program completed successfully!")
