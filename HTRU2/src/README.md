# src/

La cartella _src/_ (abbreviazione di source) viene utilizzata per contenere il **codice sorgente** di un progetto Python. È una best practice nei progetti ben strutturati per mantenere il codice organizzato e **separato** da **altri file** come _documentazione, dati e test_.

---

# Creazione di moduli python

Se vogliamo creare dei pacchetti dobbiamo creare una cartella e creare in essa il file `__init__.py_`  per renderla un modulo.
- generalmente è vuoto ma possiamo definire degli input automatici
- si possono importare delle funzioni dei file appena creati in questo modo:
```python
from .nome_file import nome_funzione
```

## Workflow per la creazione di un modulo:

1. Accedi alla cartella `src/`
2. crea una nuova cartella (il modulo) e accedi
3. crea nuovi file di script python nella nuova cartella
4. crea file `__init.py__` nella nuova cartella
    1. importa tutte/alcune funzioni con `from .nomefile import *` 
    2. aggiungi ulteriori input automatici
5. nel file di entry point è ora possibile importare il nuovo modulo tramite `from src import nome_modulo` 

Esempio di workflow:
1. Accedo a `src/`
2. Creo cartella `MyModule` ed accedo
4. Creo file `__init.py__`
5. Creo file `script.py`
6. In `script.py` aggiungo una funzione `def function()`
7. In `__init__.py` aggiungo `from .script import function` 
8. Nel file di entry point ora posso:
    - importare il nuovo modulo: `from src import MyModule` 
    - utilizzare la funzione: `MyModule.function()`

# Lista moduli:

1. models/ - 
2. utils/ - 

---