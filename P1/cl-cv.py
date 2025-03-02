import numpy as np
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# Selección de datasets
dataset_names = ['scene', 'Corel5k', 'bibtex', 'enron', 'rcv1subset5']

# Modelos a evaluar
models = {
    'Binary Relevance (RF)': BinaryRelevance(classifier=RandomForestClassifier(n_estimators=100, random_state=42)),
    'Classifier Chain (NB)': ClassifierChain(classifier=GaussianNB()),
    'Label Powerset (SVC)': LabelPowerset(classifier=SVC())
}

# Validación cruzada KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluación
results = {}
for dataset in dataset_names:
    print(f"Evaluando dataset: {dataset}")
    X, Y, _, _ = load_dataset(dataset, 'undivided')
    X = X.toarray() if hasattr(X, 'toarray') else X
    Y = Y.toarray() if hasattr(Y, 'toarray') else Y
    X = StandardScaler().fit_transform(X)
    
    results[dataset] = {}
    
    for model_name, model in models.items():
        accuracies = []
        hamming_losses = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            
            accuracies.append(accuracy_score(Y_test, Y_pred))
            hamming_losses.append(hamming_loss(Y_test, Y_pred))
        
        results[dataset][model_name] = {
            'Accuracy': np.mean(accuracies),
            'Hamming Loss': np.mean(hamming_losses)
        }
        
# Mostrar resultados
tab = "-" * 50
for dataset, model_results in results.items():
    print(f"\n{tab}\nResultados para dataset: {dataset}\n{tab}")
    for model_name, metrics in model_results.items():
        print(f"{model_name}:\n  Accuracy: {metrics['Accuracy']:.4f}\n  Hamming Loss: {metrics['Hamming Loss']:.4f}\n")