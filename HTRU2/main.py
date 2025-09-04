# Un file di entry point è il punto di ingresso principale di 
# un programma Python. È lo script che viene eseguito 
# per avviare l’applicazione o eseguire una funzionalità specifica

from src.models import train_model, function_1
from src import utils


if __name__ == "__main__":
    print ("Avvio progetto...")
    utils.function()