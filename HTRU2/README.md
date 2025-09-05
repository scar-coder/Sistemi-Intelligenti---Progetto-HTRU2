
# Classificazione pulsar e non-pulsar con il dataset HTRU2

Il progetto utilizza il dataset HTRU2 per sviluppare un modello di machine learning in grado di classificare segnali radio come pulsar o non-pulsar.

# Descrizione del progetto 

Il progetto mira a sviluppare un modello di machine learning per **classificare** segnali radio come **pulsar** o **non-pulsar** utilizzando il dataset HTRU2. Le pulsar sono stelle di neutroni altamente magnetizzate che **emettono** fasci di **radiazioni elettromagnetiche**, e la loro **identificazione** è **cruciale** per studi astrofisici. Il dataset **HTRU2** contiene **features** estratte da **segnali radio**, e il progetto include **fasi** di **preprocessing**, **selezione delle features**, **addestramento** del modello e **valutazione** delle prestazioni.
Verrà usato Python con librerie come **pandas**, **scikit-learn** e **TensorFlow/Keras** per implementare il modello di classificazione.

Il dataset HTRU2 è un unico set di dati contenete 1.639 esempi di pulsar e 16.259 di non-pulsar etichettati manualmente, per un totale di 17.898 esempi classificati con la variabile target binaria “Class”.
Ogni riga elenca prima le variabili, e l'etichetta di classe è l'ultima voce. Le etichette di classe utilizzate sono 0 (negativa) e 1 (positiva). I dati non contengono informazioni posizionali o altri dettagli astronomici. Sono semplicemente dati di features estratti da file candidati utilizzando lo strumento PulsarFeatureLab.

Tabella delle variabili:
| Nome variabile    | Ruolo   | Tipo       | Descrizione                                 | Unità   | Valori mancanti |
|-------------------|---------|------------|----------------------------------------------|---------|-----------------|
| Profile_mean      | Feature | Continuous | Forma dell’impulso                          |         | no              |
| Profile_stdev     | Feature | Continuous | Dispersione dell’impulso                    |         | no              |
| Profile_skewness  | Feature | Continuous | Punta o coda dell’impulso                   |         | no              |
| Profile_kurtosis  | Feature | Continuous | Simmetria dell’impulso                      |         | no              |
| DM_mean           | Feature | Continuous | Forma della curva DM-SNR                    |         | no              |
| DM_stdev          | Feature | Continuous | Dispersione della curva DM-SNR              |         | no              |
| DM_skewness       | Feature | Continuous | Punta o coda della curva DM-SNR             |         | no              |
| DM_kurtosis       | Feature | Continuous | Simmetria della curva DM-SNR                |         | no              |
| class             | Target  | Binary     | Classi “pulsar” - “non-pulsar”              | {0,1}   | no              |

Le prime 8 feature sono variabili continue, estratte da due tipi di curve:
1.	Integrated pulse profile (profilo integrato): una media del segnale radio pe-riodico osservato nel dominio tempo/frequenza.

2.	DM-SNR curve (dispersion measure – signal-to-noise ratio): rappresenta l’intensità del segnale al variare del valore di dispersione (DM), che corregge l’effetto “smearing” delle onde radio (effetto sfocatura che riduce il rapporto se-gnale/rumore (SNR), rendendo più difficile la classificazione) attraverso il mezzo interstellare (plasma con elettroni liberi), rendendo così l'impulso più stretto e netto, quindi più facilmente rilevabile.

Le prime 4 features (1 – 4) descrivono la forma statistica del segnale integrato in base al tempo e alla frequenza: 
1.	Profile_mean – media del profilo integrato (quanto è “centrato”);
2.	Profile_stdev – deviazione standard del profilo integrato (quanto è “sparso”);
3.	Profile_kurtosis – excess kurtosis del profilo integrato (quanto è piatta o picca-ta);
4.	Profile_skewness – skewness del profilo integrato (quanto è simmetrico o me-no).

Le ultime 4 features (5 - 8), in modo analogo, riassumono statisticamente la forma della curva DM-SNR per migliorare il rapporto segnale-rumore al variare del va-lore di DM con cui si tenta la correzione dello smearing: 
5.	DM_mean – media della curva DM-SNR (forma);
6.	DM_stdev – deviazione standard della curva DM-SNR (dispersione);
7.	DM_kurtosis – excess kurtosis della curva DM-SNR (punta o coda);
8.	DM_skewness – skewness della curva DM-SNR (asimmetria).

Queste misure permettono di distinguere i dati in:
- pulsar reale → curva DM-SNR con picco evidente e statistiche anomale,
- rumore o RFI → curva senza picchi, con distribuzioni più piatte o casuali.


# Indice 

1. [Struttura del progetto](#struttura-del-progetto)
2. [Come installare ed eseguire il progetto](#come-installare-ed-eseguire-il-progetto)
3. [Come usare il progetto](#come-usare-il-progetto)
4. [Risultati e metriche](#risultati-e-metriche)
5. [Ringraziamenti e riferimenti](#ringraziamenti-e-riferimenti)
6. [Autore e licenza](#autore-e-licenza)

# Come installare ed eseguire il progetto 

1. clonare il repository
   ```bash
   git clone https://github.com/scar-coder/Sistemi-Intelligenti---Progetto-HTRU2.git
   ```
2. navigare nella cartella del progetto
   ```bash
    cd HTRU2
    ```
3. creare e attivare un ambiente virtuale (opzionale ma consigliato)
   ```bash
    python -m venv env_htru2
    venv/bin/activate  # Su Linux usa `source venv\Scripts\activate`
   ```
4. installare le dipendenze
   ```bash
    pip install -r requirements.txt
   ```
5. Solo a fine lavoro: esegui da terminale `deactivate` per uscire dall'ambiente virtuale.



# Come usare il progetto 

esegui gli script in questa sequenza:

src/utils/1_read.py
src/utils/2_preprocessing.py
src/utils/3_train.py
src/utils/4_test.py

---

# Risultati e metriche


---

# Ringraziamenti e riferimenti 

**RINGRAZIAMENTI**:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/372/htru2

- Il professore Gianluca Zaza e la professoressa Gabriella Casalino per il supporto e per averci aiutati a strutturare il progetto


**RIFERIMENTI**:

DOI del paper di riferimento dal quale è stato ottenuto il dataset:
- R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656

DOI del dataset:
- R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.


Acknowledgements:

This data was obtained with the support of grant EP/I028099/1 for the University of Manchester  Centre for Doctoral Training in Computer Science, from the UK Engineering and Physical Sciences Research Council (EPSRC). The raw observational data was collected by the High Time Resolution Universe Collaboration using the Parkes Observatory, funded by the Commonwealth of Australia and managed by the CSIRO.

Paper che dimostra l'identificazione delle pulsar con poche features:
- Lin, H., Li, X., & Luo, Z. (2020). Pulsars Detection by Machine Learning with Very Few Features. ArXiv, abs/2002.08519.

---

# Autore e licenza

Autori del progetto:
- Scarpino Danilo
- Balestra Cristina
- Schiavo Marika


