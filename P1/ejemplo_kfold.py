import numpy as np
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss, accuracy_score

# Load dataset
X_train, y_train, _, _ = load_dataset('emotions', 'train')
print("Dataset X_train type:", type(X_train))
print("Dataset shape:", X_train.shape if hasattr(X_train, 'shape') else "No shape attribute")

X_train_scaled = StandardScaler().fit_transform(X_train.toarray())

clf = BinaryRelevance(classifier=RandomForestClassifier(n_estimators=100, random_state=42))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
hamming_losses = []

for train_idx, test_idx in kf.split(X_train_scaled):
    X_train_fold, X_test_fold = X_train_scaled[train_idx], X_train_scaled[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

    clf.fit(X_train_fold, y_train_fold)
    y_pred = clf.predict(X_test_fold)

    accuracies.append(accuracy_score(y_test_fold, y_pred))
    hamming_losses.append(hamming_loss(y_test_fold, y_pred))

print("Accuracy promedio:", np.mean(accuracies))
print("Hamming Loss promedio:", np.mean(hamming_losses))

