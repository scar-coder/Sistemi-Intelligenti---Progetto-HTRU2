# Un file di entry point è il punto di ingresso principale di 
# un programma Python. È lo script che viene eseguito 
# per avviare l’applicazione o eseguire una funzionalità specifica

from src.models import KNN, Decision_Tree, Random_Forest
from src.utils.read import *
from src.utils.preprocess import *
from src.utils.train import *
from src.utils.test import *


if __name__ == "__main__":
    print ("Caricamento detaset...")
    
    dataset = load_dataset()
    dataset_info(dataset)
    
    print("Preprocessing del dataset...")
    