VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# The setup target depends on the prerequisite `requirements.txt`,
# whenever the file changes, the dependencies will be refreshed.
venv/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	$(PYTHON) run.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)