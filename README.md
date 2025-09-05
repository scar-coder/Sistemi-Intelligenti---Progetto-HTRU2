# Machine learning project template

Template per un progetto di machine learning organizzato e facilmente manutenibile.  

# Indice
1. [Descrizione struttura del template](#descrizione-template) 
2. [Gestione README.md](#gestione-readmemd) 
3. [Gestione ambiente virtuale](#gestione-ambiente-virtuale)
4. [Gestione librerie python da terminale](#gestione-librerie-python-da-terminale)
5. [Google Colab](#google-colab)
6. [Come usare il progetto](#come-usare-il-progetto)
	1. [Workflow di personalizzazione del template](#workflow-di-personalizzazione-del-template)
	2. [Workflow di creazione di un modulo python](#workflow-di-creazione-di-un-modulo-python)
7. [Struttura Progetto](#struttura-progetto)
	1. [makefile](#makefile)
	2. [requirements.txt](#requirementstxt)
	3. [setup.py](#setuppy)
	4. [Entry point file](#entry-point-file)
	5. \_\_[init__.py](#__init__py)
	6. [src/](#src)
	7. [out/](#out)
	8. [data/](#data)
	9. [docs/](#docs)
	10. [notebooks/](#notebooks)
8. [Ringraziamenti e riferimenti](#ringraziamenti-e-riferimenti)


# Descrizione template

Questo documento mostra le istruzioni su come usare questo template per creare un progetto di machine learning in modo organizzato e gestirlo in modo efficace. La struttura del template è progettata per agevolare:
- Scalabilità e Modularità 
- Performance e Efficienza Computazionale 
- Collaborazione e Condivisione
- Manuntenibiltà e Chiarezza del Codice 
- Riproducibilità e Affidabilità


Struttura del progetto:
```
📂 Machine Learning Project Template
├── 📂 data                     # Cartella per la gestione dei dataset
│   ├── 📂 raw                  # Dati grezzi non elaborati
│   ├── 📂 interim              # Dati intermedi elaborati
│   ├── 📂 processed            # Dati finali pronti per l'uso
│   └── 📄 README.md            # Documentazione per la cartella data
├── 📂 docs                     # Documentazione aggiuntiva
│   └── 📄 README.md            # Dettagli sulla documentazione
├── 📂 notebooks                # Notebook Jupyter per analisi e sviluppo
│   ├── 📄 notebook.ipynb       # Notebook di esempio
│   └── 📄 README.md            # Documentazione per i notebook
├── 📂 src                      # Codice sorgente del progetto
│   ├── 📂 models               # Modelli di machine learning
│   │   ├── 📄 __init__.py      # Inizializzazione del pacchetto models
│   ├── 📂 utils                # Utility e funzioni di supporto
│   │   ├── 📄 __init__.py      # Inizializzazione del pacchetto utils
│   └── 📄 README.md            # Documentazione per il codice sorgente
├── 📄 __init__.py              # File per definire il pacchetto principale
├── 📄 LICENSE                  # Licenza del progetto
├── 📄 main.py                  # Entry point del progetto
├── 📄 makefile                 # Automazione delle operazioni comuni
├── 📄 project workflow.ipynb        # Notebook per eseguire il workflow del progetto
├── 📄 README.md                # Documentazione principale del progetto
├── 📄 requirements.txt         # Dipendenze del progetto
├── 📄 setup.ipynb              # Notebook per la distribuzione del pacchetto
├── 📄 setup.py                 # Script per la creazione e distribuzione del pacchetto

```

## Gestione README.md

Ogni sezione del progetto ha un file README.md che ne descrive lo scopo, tranne quello della cartella principale, che fa parte del template e documenta e guida l’uso di un progetto, rendendo il codice comprensibile e accessibile a sviluppatori e ricercatori 

Struttura del README principale del progetto: 
1. Titolo e Descrizione Breve 
2. Descrizione del progetto 
3. Indice 
4. Come installare ed eseguire il progetto 
5. Come usare il progetto
6. Risultati e metriche 
7. Ringraziamenti e riferimenti 
8. Autore e licenza.


---

# Gestione ambiente virtuale

Un ambiente virtuale (virtual environment) in Python permette di **isolare** le **dipendenze** di un progetto, **evitando** **conflitti** con altre installazioni di librerie nel sistema:

1. Apri il Terminale o il Prompt dei Comandi
    - Assicurati di essere nella directory principale del tuo progetto o di scegliere una directory appropriata in cui creare il tuo ambiente virtuale
1. Esegui `python -m venv env_progetto` (esempio di nome: env_progetto)
    - crea una directory env_progetto nella tua directory corrente. All’interno di questa directory, verranno creati i file e le cartelle necessari per l’ambiente virtuale
3. Attiva l’Ambiente Virtuale
    - Linux: `source env_progetto/bin/activate`
    - Windows: `env_progetto\Scripts\activate`
4. Solo a fine lavoro: esegui da terminale `deactivate` per uscire dall'ambiente virtuale.

---

# Gestione librerie python da terminale

Installare una libreria:
```
pip install nome_libreria
``` 
Installare una versione specifica:
```
pip install nome_libreria==1.1.0
``` 
Aggiornare una libreria:
```
pip --upgrade nome_libreria
``` 
Disinstallare una libreria:
```
pip uninstall nome_libreria
```


---

# Google Colab

Funzionalità utili:
Installare moduli aggiuntivi:
`! pip install nome_modulo`

Clonare un repository Git:
`! git clone https://github.com/user/repo`

Interagire con Google Drive:
```python
from google.colab import drive
drive.mount('/my-drive')
```


---

# Come usare il progetto

1. Crea una nuova cartella
2. clona questo template con `git clone https://github.com/scar-coder/ML-project-template.git`

Dal notebook `project workflow.ipynb` (utile per [Google Colab](# Google Colab))
1. Scarica il notebook `project workflow.ipynb` in una cartella dedicata
2. Apri il notebook `project workflow.ipynb`
3. Importare i file del progetto col comando presente in "Clona progetto"

## Workflow di personalizzazione del template

 (una volta sola) **Aggiungere le informazioni iniziali del progetto**:
- Cambia il nome della cartella principale 
- Aggiungi nel `README.md`
	- Titolo e breve descrizione 
	- Prima descrizione dettagliata se possibile
		- suggerimento: usare tecnica delle 5W (Chi/Cosa, Come, Quando, Dove, Perché)
	- Autore e primi riferimenti
- Aggiungi una licenza (auto-compilato con github durante la creazione di un repo)
- (opzionale) aggiorna il parametro VENV nel  [makefile](##makefile)
- (opzionale) aggiungi le informazioni su autore e progetto in [setup.py](##setup.py)


> [!TIP]
> Se 'make' non funziona
> Installare make tramite [choco](https://chocolatey.org/install):
> `choco install make`
>
> fonte: [stackoverflow: How to run a makefile in windows](https://stackoverflow.com/questions/2532234/how-to-run-a-makefile-in-windows#:~:text=You%20can%20install%20GNU,Run%20choco%20install%20make)


1. **Modificare** README.md **principale**:
    - Aggiorna descrizione specifica del tuo progetto e aggiorna l'indice se necessario
2. **Personalizzare la cartella** `src/`:
    - Aggiungi script Python o moduli nei sottopacchetti `models` e `utils`
    - Ricorda di aggiornare i file `__init__.py` per facilitare gli import
    - [workflow per creare un nuovo modulo python](#Workflow di creazione di un modulo python):
	- Aggiorna la `versione` del progetto nel file [setup.py](##setup.py)
3. **Aggiungere nuovi notebook**:
    - Inserisci i tuoi notebook nella cartella `notebooks/` 
    - aggiorna `README.md` interno
4. **Gestione dei dati**:
    - Inserisci i tuoi dataset in `data/raw`
    - Pre-elabora i dati in `data/interim` 
    - Processa i dati pre-elaborati in `data/processed`
5. **Configurare le dipendenze**:
    - Aggiungi librerie mancanti al file [requirements.txt](##requirements.txt) tramite `pip freeze > requirements.txt` 
    - aggiornare il parametro `requisiti_pacchetti` nel file [setup.py](##setup.py)
6. **Automatizzare task** nel **[Makefile](##makefile)**:
    - Inserisci comandi frequenti: 
	    - esecuzione test, 
	    - setup ambiente, 
	    - linting, ecc.
7. **Distribuire il pacchetto** con [setup.py](##setup.py) (opzionale):
    - Rivedi il file `setup.py` con i tuoi dati prima di procedere
    - se necessario, carica su PyPI il pacchetto utilizzando il notebook `setup.ipynb`

## Workflow di creazione di un modulo python

1. Accedi alla cartella [src/](##src/) 
2. crea una nuova cartella (il modulo) e accedi
3. crea nuovi file di script python nella nuova cartella
4. crea file [__init.py__](##\_\_init__.py) nella nuova cartella per renderlo un modulo python
	1. importa tutte/alcune funzioni con `from .nomefile import *` 
	2. aggiungi ulteriori input automatici
5. nel [file di entry point](##Entry point file) è possibile importare il nuovo modulo tramite `from src import nome_modulo` 

Esempio di workflow:
1. Accedo a `src/`
2. Creo cartella `MyModule` ed accedo
3. Creo file `__init.py__`
4. Creo file `script.py`
5. In `script.py` creo una funzione `def function()`
6. In `__init__.py` importo la funzione: `from .script import function`
7. Nel file di entry point ora posso:
	- importare il nuovo modulo: `from src import MyModule`
	- utilizzare la funzione: `MyModule.function()`


---

# Struttura Progetto

## makefile

Il file **Makefile** viene utilizzato per **automatizzare** e **semplificare** l’esecuzione di **comandi ripetitivi** in un progetto. 

È molto utile nei progetti di Machine Learning, Deep Learning e sviluppo software per eseguire **compilazioni**, **test**, **esecuzioni** e **pulizie** del codice con un semplice comando 

Perché è utilizzarlo ?
- Automatizza attività ripetitive (es. installazione, training, testing) 
- Evita errori umani. Un solo comando garantisce che tutti i passaggi vengano eseguiti correttamente 
- Standardizza il workflow. Perfetto per team di sviluppo, evitando configurazioni manuali diverse. 
- Compatibilità con Linux/macOS. Funziona con GNU Make, molto usato in ambienti Unix

## requirements.txt

Il file requirements.txt è utilizzato nei progetti Python per **gestire** e **installare** le **dipendenze** in modo rapido e riproducibile. 

Contiene l’elenco delle librerie necessarie per eseguire il progetto, spesso con versioni specifiche per evitare problemi di compatibilità 
1. Installare le dipendenze: `pip install -r requirements.txt` 
2. Creare un «requirements.txt» da un ambiente Python: `pip freeze > requirements.txt`

## setup.py

Il file setup.py è il file di configurazione principale per la **creazione**, **distribuzione** e **installazione** di **pacchetti** Python. 

Viene utilizzato per **trasformare** il **codice** Python in un **pacchetto** installabile tramite **pip**, rendendolo **riutilizzabile** e **distribuibile** su PyPI (Python Package Index) o in ambienti aziendali.
Aprire il terminale ed eseguire i seguenti comandi 
1. Creazione della cartella dist/ `python setup.py sdist bdist_wheel` 
2. Se vogliamo caricare il pachetto su PyPI 
	```
	pip install twine 
	twine upload dist/*
	``` 
3. Installare il pachetto `pip install nome_pachetto`

## Entry point file

Un file di entry point è il punto di ingresso principale di un programma Python. È lo script che viene eseguito per avviare l’applicazione o eseguire una funzionalità specifica


## \_\_init__.py

Il file \_\_init__.py indica a Python che una cartella deve essere trattata come un modulo o un pacchetto. Senza questo file, Python non riconoscerebbe la cartella come parte del progetto e non permetterebbe di importarne i moduli.

Se vogliamo creare dei pacchetti dobbiamo creare una cartella e creare in essa il file `__init__.py`.
- generalmente è vuoto ma possiamo definire degli input automatici
- si possono importare delle funzioni dei file appena creati in questo modo:
```python
from .nome_file import nome_funzione
```
In questo modo posso richiamare le funzioni create con molta facilità all’interno del [file di entry point](## Entry point file)

## src/

La cartella _src/_ (abbreviazione di source) viene utilizzata per contenere il **codice sorgente** di un progetto Python. È una best practice nei progetti ben strutturati per mantenere il codice organizzato e **separato** da **altri file** come _documentazione, dati e test_.

## out/

La cartella _out/_ (abbreviazione di "**output**") viene utilizzata per salvare i **risultati** generati dal programma, come _modelli addestrati, file di log, report, immagini o previsioni_. E' buona pratica avere una sottocartella per ogni tipo di output.

Sottocartelle:
1. figures/ - Contiene le immagini
2. logs/ - Contiene i file di log
3. models/ - Contiene i modelli addestrati
4. results/ - Contiene i report e/o previsioni

## data/

La cartella _data/_ serve per **archiviare**, **organizzare** e **gestire** i **dataset** utilizzati in un progetto di Machine Learning, Deep Learning o Data Science. Una buona organizzazione aiuta a mantenere il progetto _pulito, evitare errori e garantire riproducibilità_.

Si compone di tre sottocartelle:
1. raw/ - Dati Grezzi:
	- Dati originali, non modificati. Qui si salvano i file così come vengono raccolti o scaricati
2. interim/ - Dati Intermedi:
	- Dati parzialmente elaborati durante il preprocessing. Qui si trovano dataset ripuliti ma non ancora finalizzati. Possono contenere file con dati filtrati, normalizzati o con report di qualità
3. processed/ - Dati Pronti per il Modello:
	- Dati completamente elaborati e pronti per l'addestramento. Qui si salvano i dataset finali dopo

## docs/

La cartella _docs/_ è utilizzata per archiviare la **documentazione** del progetto. È fondamentale per aiutare gli sviluppatori e i collaboratori a **comprendere** _il codice, le funzionalità e il funzionamento_ dell'**applicazione** o del **modello** di Machine Learning.


## notebooks/

La cartella _notebooks/_ viene utilizzata nei progetti di Machine Learning, Deep Learning e Data Science per **contenere** e **organizzare** i **Notebooks Jupyter**(.ipynb), file interattivi essenziali per _l'esplorazione dei dati, la prototipazione, la visualizzazione e l'analisi dei risultati_. Questi file permettono di **scrivere codice** eseguibile, **visualizzare output** e **integrare testo** descrittivo in un unico documento. Possono essere _aperti e modificati_ con **Jupyter Notebook** o **JupyterLab** e sono ampiamente usati anche nello **sviluppo Python**

Nel template sono presenti i seguenti notebooks:
1. `project workflow.ipynb`: notebook per eseguire il workflow del progetto (usa makefile)
2. `setup.ipynb`: notebook per creare e distribuire il progetto come pacchetto installabile
3. `notebook.ipynb`: un notebook vuoto di esempio


---

# Ringraziamenti e riferimenti

- [Dott. Gianluca Zaza](https://sites.google.com/site/cilabuniba/people/gianluca-zaza), Università degli Studi di Bari
- [Danilo Scarpino (scar-coder)](https://github.com/scar-coder)
- Documentazione ufficiale Python: https://docs.python.org/
