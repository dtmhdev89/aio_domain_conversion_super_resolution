# Variables
PYTHON_VERSION := $(shell python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
PYTHON = python3
PIP = pip3
VENV_NAME = .venv
VENV_BIN = $(VENV_NAME)/bin

# Create virtual environment
install_venv:
	ifeq ($(PYTHON_VERSION),3.7)
        apt-get install -y python3.7-venv
	else ifeq ($(PYTHON_VERSION),3.8)
			apt-get install -y python3.8-venv
	else ifeq ($(PYTHON_VERSION),3.9)
			apt-get install -y python3.9-venv
	else ifeq ($(PYTHON_VERSION),3.10)
			apt-get install -y python3.10-venv
	else ifeq ($(PYTHON_VERSION),3.11)
			apt-get install -y python3.11-venv
	else ifeq ($(PYTHON_VERSION),3.12)
			apt-get install -y python3.12-venv
	else
			@echo "Unsupported Python version: $(PYTHON_VERSION). Please install python3.x-venv manually."
			exit 1
	endif

venv:
	$(PYTHON) -m venv $(VENV_NAME)

# Library installation if not install by requirements.txt
libs_install:
	$(VENV_BIN)/$(PIP) install gdown memory_profiler numpy \
		torch torcheval torchsummary torchvision

# Install dependencies by requirements.txt. This requires specific python version (3.12)
install: requirements.txt
	$(VENV_BIN)/$(PIP) install -r requirements.txt

# Export required libraries
export_requirements:
	$(VENV_BIN)/$(PIP) freeze > requirements.txt

# Clean up
clean:
	@if [-d "$(VENV_NAME)"]; then \
		rm -rf $(VENV_NAME); \
		echo "Remove $(VENV_NAME);" \
	else \
		echo "$(VENV_NAME) not found, nothing to remove."; \
	fi

# Download dataset
download_dataset:
	$(VENV_BIN)/gdown --id 1bsWkNmmYvBrgE1c58SGJFcCjQv3SUyH3 -O Khoa_LHR_image.zip
	unzip -d . Khoa_LHR_image.zip
	rm Khoa_LHR_image.zip

# Model
run_unet_model: unet_model.py
	$(VENV_BIN)/$(PYTHON) unet_model.py

.PHONY: venv

help:
	@echo "Usage: make <target> [OPTIONS]"
	@echo ""
	@echo "Description: This Makefile provides targets for running projects"
	@echo ""
	@echo "Available Targets:"
	@echo "venv:"
	@echo "		Description: Create .venv for environment"
	@echo "		Usage: make venv"
