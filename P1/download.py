import numpy as np
import pandas as pd
import requests
import os
from scipy.stats import chi2_contingency

def download_dataset(url, file_path):
    """ Descarga un dataset desde una URL y lo guarda en un archivo local. """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Dataset descargado correctamente: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el dataset desde {url}: {e}")

def check_label_distribution(Y):
    """Verifica si hay etiquetas con valores constantes."""
    constant_labels = []
    for j in range(Y.shape[1]):
        unique_values = np.unique(Y[:, j])
        if len(unique_values) == 1:
            constant_labels.append(j)
    return constant_labels

def compute_metrics(X, Y):
    """
    Calcula métricas sobre el dataset, evitando errores en chi2_contingency.
    """
    n = X.shape[0]
    f = X.shape[1]
    l = Y.shape[1]
    
    car = Y.sum() / n
    den = car / l
    
    unique_labelsets = np.unique(Y, axis=0)
    num_unique = unique_labelsets.shape[0]
    total_possible = 2 ** l
    div = (num_unique / total_possible) * 100  # en porcentaje
    
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
            
            # Evitar tablas con celdas en 0
            if np.any(table == 0):
                continue
            
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

def load_csv_dataset(file_path, label_columns, sep):
    """
    Carga un dataset desde un archivo CSV con detección de delimitadores.
    """
    df = pd.read_csv(file_path, sep=sep, quoting=3, encoding='utf-8', on_bad_lines='skip')
    print(df.head())  # Verificar estructura del dataset
    
    # Asegurar que las columnas de etiquetas existen
    for col in label_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataset")
    
    X = df.drop(columns=label_columns).values
    Y = df[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values
    
    # Verificar etiquetas con valores constantes
    constant_labels = check_label_distribution(Y)
    if constant_labels:
        print(f"Advertencia: Se encontraron etiquetas constantes en las columnas {constant_labels}. Se excluirán del análisis.")
        Y = np.delete(Y, constant_labels, axis=1)
    
    return X, Y

if __name__ == '__main__':
    # Definir los datasets con sus URLs y las columnas de etiquetas
    datasets = {
        'MultiLabel_Classification': {
            'url': 'https://huggingface.co/datasets/owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH/resolve/main/PubMed%20Multi%20Label%20Text%20Classification%20Dataset%20Processed.csv',
            'file': 'multilabel_classification.csv',
            'sep': ',',
            'label_columns': ['Title', 'abstractText', 'meshMajor', 'pmid', 'meshid', 'meshroot', 
                              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']
        },
        'MultiLabel_Movie_Classification': {
            'url': 'https://raw.githubusercontent.com/shashankvmaiya/Movie-Genre-Multi-Label-Text-Classification/master/Data/movies_genres.csv',
            'file': 'multilabel_movie_classification.csv',
            'sep': '\t',
            'label_columns': ['title', 'plot', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 
                              'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show', 
                              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 
                              'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western', 'plot_lang']
        }
    }


    
    results = {}
    for name, info in datasets.items():
        print(f"Procesando dataset: {name}")
        try:
            download_dataset(info['url'], info['file'])
            X, Y = load_csv_dataset(info['file'], info['label_columns'], info['sep'])
            metrics = compute_metrics(X, Y)
            results[name] = metrics
            os.remove(info['file'])  # Eliminar archivo después de procesarlo
        except Exception as e:
            print(f"Error al procesar {name}: {e}")
    
    # Mostrar resultados
    for ds, metrics in results.items():
        print("\nDataset:", ds)
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("-" * 50)