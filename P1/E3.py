#!/usr/bin/env python

import numpy as np
from scipy.stats import chi2_contingency
import scipy.sparse

# Para cargar los datasets de scikit-multilearn
from skmultilearn.dataset import load_dataset

def compute_metrics(X, Y):
    """
    Calcula las medidas de caracterización:
      - n: número de instancias
      - f: número de atributos
      - l: número de etiquetas
      - car: cardinality (promedio de etiquetas positivas por instancia)
      - den: density (car / l)
      - div: diversity = (número de combinaciones únicas de etiquetas / 2^l) * 100
      - avgIR: promedio de imbalance ratio por etiqueta
      - rDep: proporción de pares de etiquetas dependientes (p < 0.01 en chi-cuadrado)
    """
    if scipy.sparse.issparse(Y):
        Y = Y.toarray()
    if scipy.sparse.issparse(X):
        X = X.toarray()
    
    n = X.shape[0]
    f = X.shape[1]
    l = Y.shape[1]
    
    car = Y.sum() / n
    den = car / l
    
    unique_labelsets = np.unique(Y, axis=0)
    num_unique = unique_labelsets.shape[0]
    total_possible = 2 ** l
    div = (num_unique / total_possible) * 100 
    
    ir_list = []
    for j in range(l):
        positives = Y[:, j].sum()
        if positives == 0:
            continue 
        ir = (n - positives) / positives
        ir_list.append(ir)
    avgIR = np.mean(ir_list) if ir_list else np.nan
    
    dependent_pairs = 0
    total_pairs = 0
    for i in range(l):
        for j in range(i + 1, l):
            total_pairs += 1
            a = np.sum((Y[:, i] == 1) & (Y[:, j] == 1))
            b = np.sum((Y[:, i] == 1) & (Y[:, j] == 0))
            c = np.sum((Y[:, i] == 0) & (Y[:, j] == 1))
            d = np.sum((Y[:, i] == 0) & (Y[:, j] == 0))
            table = np.array([[a, b], [c, d]])
            chi2, p, _, _ = chi2_contingency(table, correction=False)
            if p < 0.01:
                dependent_pairs += 1
    rDep = dependent_pairs / total_pairs if total_pairs > 0 else np.nan

    return {
        'n': n,
        'f': f,
        'l': l,
        'cardinality': car,
        'density': den,
        'diversity (%)': div,
        'avgIR': avgIR,
        'rDep': rDep
    }

def process_dataset(dataset_name):
    print("Procesando dataset:", dataset_name)
    try:
        X, Y, _, _ = load_dataset(dataset_name, 'undivided')
        metrics = compute_metrics(X, Y)
        return metrics
    except Exception as e:
        print(f"Error al procesar {dataset_name}: {e}")
        return None

if __name__ == '__main__':
    dataset_names = ['scene', 'Corel5k', 'bibtex', 'enron', 'rcv1subset5', 'tmc2007_500',
                     'rcv1subset3', 'rcv1subset1', 'delicious', 'rcv1subset4', 'genbase', 
                     'birds', 'emotions', 'rcv1subset2', 'mediamill', 'medical', 'yeast']
    
    results = {}
    for ds in dataset_names:
        metrics = process_dataset(ds)
        if metrics is not None:
            results[ds] = metrics
    
    # Mostrar resultados de forma ordenada
    for ds, metrics in results.items():
        print("\nDataset:", ds)
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("-" * 50)
