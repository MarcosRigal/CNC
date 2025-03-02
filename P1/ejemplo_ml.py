from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, accuracy_score

X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

br_clf = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
br_clf.fit(X_train, y_train)
br_predictions = br_clf.predict(X_test)

cc_clf = ClassifierChain(classifier=GaussianNB())
cc_clf.fit(X_train.toarray(), y_train.toarray())
cc_predictions = cc_clf.predict(X_test.toarray())

lp_clf = LabelPowerset(classifier=SVC())
lp_clf.fit(X_train, y_train)
lp_predictions = lp_clf.predict(X_test)

br_knn_clf = BinaryRelevance(classifier=KNeighborsClassifier(n_neighbors=5), require_dense=[False, True])
br_knn_clf.fit(X_train, y_train)
br_knn_predictions = br_knn_clf.predict(X_test)

print("\n--- Métricas Generales ---")
print("Binary Relevance (SVC) - Hamming Loss:", hamming_loss(y_test, br_predictions))
print("Classifier Chain (NB) - Hamming Loss:", hamming_loss(y_test, cc_predictions))
print("Label Powerset (SVC) - Hamming Loss:", hamming_loss(y_test, lp_predictions))
print("Binary Relevance (kNN) - Hamming Loss:", hamming_loss(y_test, br_knn_predictions))