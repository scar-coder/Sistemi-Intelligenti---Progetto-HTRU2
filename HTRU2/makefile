# Comandi specifici di progetto
# Prepara i dati
prepare_data:
	python src/model/preprocess.py

# Addestra il modello
train:
	python src/model/train.py --epochs 10 --batch_size 32

# Esegue i test
test:
	pytest src/tests/


#---

# Comandi generali 

# Comandi per inizializzare il progetto

# Specifica il nome dell’ambiente virtuale
VENV = venv

# Carica il progetto da github
clone:
	git clone https://github.com/scar-coder/ML-project-template.git

# Installa le dipendenze in base al sistema operativo
install_win:
	python -m venv $(VENV)
	$(VENV)\Scripts\activate
	$(VENV)\Scripts\pip install -r requirements.txt

install_linux: 
	python -m venv $(VENV) 
	$(VENV)/bin/activate 
	$(VENV)/bin/pip install -r requirements.txt


# Pulisce i file temporanei e cache
clean:
	rm -rf __pycache__ *.pyc $(VENV)

# Aiuto: mostra i comandi disponibili
help:
	@echo "Comandi disponibili:"
	@echo "  make clone			- Carica template di progetto da github"
	@echo "  make install       - Installa le dipendenze"
	@echo "  make prepare_data  - Prepara il dataset"
	@echo "  make train         - Addestra il modello"
	@echo "  make test          - Esegue i test"
	@echo "  make clean         - Pulisce i file temporanei"
