import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Paths
RAW_DATA_FOLDER = "raw_generated_data"
PROCESSED_DATA_FILE = "processed_dataset.csv"

def load_raw_datasets(raw_data_folder):
    """Loads all datasets from the raw data folder."""
    datasets = []
    for file in os.listdir(raw_data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(raw_data_folder, file)
            dataset = pd.read_csv(file_path, header=None).values
            datasets.append(dataset)
    return datasets

def extract_features(dataset):
    """Extract features of interest from a dataset."""
    # Independent variable computation
    n_rows = dataset.shape[0]
    n_classes = len(np.unique(dataset[:, 0]))
    n_features = dataset.shape[1] - 1
    noise_level = np.var(dataset[:, 1:], axis=0).mean()
    
    return [n_rows, n_classes, n_features, noise_level]

def find_best_k(dataset):
    """Runs a kNN model and finds the best hyperparameter 'k'."""
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    best_k, best_accuracy = 1, 0
    for k in range(1, min(len(y), 50)):  # Limit k to avoid large values
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_k, best_accuracy = k, accuracy
    return best_k

def process_single_dataset(dataset):
    """Traitement d'un seul dataset : extraction des features et calcul du meilleur k."""
    features = extract_features(dataset)
    best_k = find_best_k(dataset)
    print("1 dataset traité")
    return [best_k] + features

def create_dataset(raw_data_folder, output_file):
    print(f"Chemin des données : {raw_data_folder}")
    datasets = load_raw_datasets(raw_data_folder)
    print(f"{len(datasets)} datasets chargés. Lancement du traitement parallèle...")

    # Mesure du temps de démarrage
    start_time = time.time()

    # Traitement en parallèle
    with ProcessPoolExecutor() as executor:
        processed_data = list(executor.map(process_single_dataset, datasets))

    # Mesure du temps de fin
    end_time = time.time()

    # Sauvegarde des résultats
    header = ['best_k', 'n_rows', 'n_classes', 'n_features', 'noise_level']
    df = pd.DataFrame(processed_data, columns=header)
    print("Sauvegarde des résultats...")
    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")

    # Affichage du temps pris
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution (parallèle) : {elapsed_time:.2f} secondes")
    print(elapsed_time)

def create_dataset_single_thread(raw_data_folder, output_file):
    print(raw_data_folder)
    """Creates the dataset of interest with extracted features."""
    datasets = load_raw_datasets(raw_data_folder)

    # Mesure du temps de démarrage
    start_time = time.time()

    processed_data = []
    for dataset in datasets:
        features = extract_features(dataset)
        best_k = find_best_k(dataset)
        processed_data.append([best_k] + features)
        print("1 de fait")

    # Mesure du temps de fin
    end_time = time.time()

    # Sauvegarde des résultats
    header = ['best_k', 'n_rows', 'n_classes', 'n_features', 'noise_level']
    df = pd.DataFrame(processed_data, columns=header)
    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")

    # Affichage du temps pris
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution (single-thread) : {elapsed_time:.2f} secondes")
    return elapsed_time

def compare_execution_times():
    # Définir les chemins et fichiers
    RAW_DATA_FOLDER = "raw_generated_data"
    PROCESSED_DATA_FILE_PARALLEL = "processed_dataset_parallel.csv"
    PROCESSED_DATA_FILE_SINGLE = "processed_dataset_single.csv"

    # Exécution en parallèle
    print("Démarrage de l'exécution en parallèle...")
    time_parallel = create_dataset(RAW_DATA_FOLDER, PROCESSED_DATA_FILE_PARALLEL)

    # Exécution en single-thread
    print("Démarrage de l'exécution en single-thread...")
    time_single_thread = create_dataset_single_thread(RAW_DATA_FOLDER, PROCESSED_DATA_FILE_SINGLE)

    # Comparaison des temps d'exécution
    print("\n=== Comparaison des temps d'exécution ===")
    print(f"Temps parallèle     : {time_parallel:.2f} secondes")
    print(f"Temps single-thread : {time_single_thread:.2f} secondes")
    print(f"Gain de performance : {time_single_thread / time_parallel:.2f}x plus rapide (parallèle)")