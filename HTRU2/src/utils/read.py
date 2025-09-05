import pandas as pd
import os

# Percorso del file dataset


# Carica il dataset usando pandas
# Cartelle disponibili: raw, interim, processed
def load_dataset():
    path= os.path.join(os.path.dirname(__file__), '../../data/raw/HTRU_2.csv')
    
    names=[ "Profile_mean", "Profile_std", 
            "Profile_skewness", "Profile_kurtosis",
            "DM_mean", "DM_std", "DM_skewness", 
            "DM_kurtosis", "Class"]
    df = pd.read_csv(path, header=None, names=names)
    return df


# Mostra informazioni sul dataset
def dataset_info(dataset):
    print(f"Number of samples: {dataset.shape[0]}")
    print(f"Number of features: {dataset.shape[1] - 1}")  # Esclude la colonna della classe
    print(f"Class distribution: {dataset['Class'].value_counts()}") 
    print("\n---\n Data examples:")
    print(dataset.head())
    print("\n---\n Dataset summary:")
    print(dataset.info())


# esempi di utilizzo
#df = load_dataset()
#dataset_info(df)