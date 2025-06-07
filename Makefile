#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = store-sales-DL
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 store-sales-DL
	isort --check --diff store-sales-DL
	black --check store-sales-DL

## Format source code with black
.PHONY: format
format:
	isort store-sales-DL
	black store-sales-DL



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	
## Run dataset processing pipeline
.PHONY: dataset
dataset:
	$(PYTHON_INTERPRETER) -m store_sales_DL.dataset

## Run feature engineering pipeline
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m store_sales_DL.features

## Train the deep learning model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m store_sales_DL.modeling.train

## Run model inference/prediction
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m store_sales_DL.modeling.predict

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
