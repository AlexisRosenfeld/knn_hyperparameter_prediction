import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Ensure the folder exists
RAW_DATA_FOLDER = "raw_generated_data"
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

def clear_raw_data_folder():
    """Deletes all files in the RAW_DATA_FOLDER."""
    for filename in os.listdir(RAW_DATA_FOLDER):
        file_path = os.path.join(RAW_DATA_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Supprime le fichier
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def genf_multiple_datasets(count, view=False):
    print("salut")
    """
    Generates multiple fake datasets and saves them as CSV files.
    
    Parameters:
        count (int): Number of datasets to generate.
        view (bool, optional): Whether to display the plots for each dataset. Defaults to False.
    """
    # Clear the folder before generating new datasets
    print("Clearing raw data folder...")
    clear_raw_data_folder()
    print("Folder cleared. Generating datasets...")
    
    for i in range(1, count + 1):
        print(f"Generating dataset {i}/{count}...")
        genf_dataset(index=i, view=view)
    print(f"Generated {count} datasets.")

def plot_dataset(dataset, cx=None):
    """Shows the 'x' values of fake dataset as scatterplots."""
    fig, ax = plt.subplots(1, 2)
    n_rows, n_cols = dataset.shape
    lx = np.arange(1, n_rows + 1, 1)  # x-axis (row-index)
    for i in range(1, n_cols):  # y-axis (x-value)
        alpha = 0.7 if not cx else 0.4 if cx[i - 1] == 0 else 0.8
        ax[0].scatter(lx, dataset[:, i], alpha=alpha)
        alpha = 0.7 if not cx else 0.1 if cx[i - 1] == 0 else 0.8
        ax[1].scatter(dataset[:, i], dataset[:, 0], alpha=alpha)
    ax[0].set_xlabel("Row index")
    ax[0].set_ylabel("X-value")
    ax[1].set_xlabel("X-value")
    ax[1].set_ylabel("Y-value")
    fig.tight_layout()
    plt.show()

def save_dataset_to_csv(dataset, index):
    """Saves the dataset to a CSV file in the raw data folder."""
    file_path = os.path.join(RAW_DATA_FOLDER, f"g{index}.csv")
    df = pd.DataFrame(dataset)
    df.to_csv(file_path, index=False, header=False)
    print(f"Dataset saved to {file_path}")

def genf_dataset(index, view=False):
    """Generates a single fake dataset and saves it to a CSV file."""
    lds = random.randint(100, 10000)  # nb of rows
    lyc = random.randint(3, 5)  # nb of 'y' classes
    cx, ix = [], 0
    lx = random.randint(3, 9)
    lxd = random.randint(1, int(lx))
    for i in range(lx):
        c = random.randint(0, 1)
        if c == 1 and ix < lxd:
            cx.append(random.randint(1, 3))
            ix += 1
            continue
        cx.append(0)
    distr_noise = random.randint(1, 3)
    dataset = []
    for r in range(lds):
        y = random.randint(0, lyc - 1)
        yr = (y + 1) / (lyc + 1)
        noise = (lds / 20) * distr_noise
        dataset.append([y])
        for c in cx:
            match c:
                case 0:
                    x = random.randint(0, 1000)
                case 1:
                    x = np.random.normal(1000 * yr, noise)
                case 2:
                    x = np.random.normal(1000 * (yr ** 2), noise)
                case 3:
                    x = random.randint(0, 1000) if random.randint(0, 100) > 80 else np.random.normal(1000 * yr, noise)
            x = 0 if x < 0 else 1000 if x > 1000 else x
            dataset[-1].append(x)
    dataset = np.array(dataset)
    save_dataset_to_csv(dataset, index)  # Save the dataset
    if view:
        plot_dataset(dataset, cx)
    return dataset