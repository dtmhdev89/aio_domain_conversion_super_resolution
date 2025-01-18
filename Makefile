# Variables
PYTHON_VERSION := $(shell python3 --version | awk '{print $$2}' | cut -d. -f1,2)
VENV_PACKAGE := python$(PYTHON_VERSION)-venv
PYTHON = python3
PIP = pip3
VENV_NAME = .venv
VENV_BIN = $(VENV_NAME)/bin
## Using specific python version if install with requirements.txt
PYTHON_VERSION_REQUIRED = 3.12

# Create virtual environment
install_venv:
	@echo "Detected Python version: $(PYTHON_VERSION)"
	@echo "Installing package: $(VENV_PACKAGE)"
	apt-get install -y $(VENV_PACKAGE)

venv: install_venv
	$(PYTHON) -m venv $(VENV_NAME)

# Library installation if not install by requirements.txt
libs_install:
	$(VENV_BIN)/$(PIP) install gdown matplotlib memory_profiler numpy \
		matplotlib memory_profiler numpy \
		ipykernel opencv-python \
		torch torcheval torchsummary torchvision

	@if pgrep -af "jupyter-notebook" > /dev/null || pgrep -af "ipykernel" > /dev/null; then \
		echo "Running in a Jupyter notebook environment"; \
		$(VENV_BIN)/$(PIP) install ipykernel; \
	else \
		echo "Not running in a Jupyter notebook environment"; \
	fi

# Install dependencies by requirements.txt. This requires specific python version (3.12)
install: requirements.txt check-python-version
	$(VENV_BIN)/$(PIP) install -r requirements.txt

check-python-version:
	@if [ "$(PYTHON_VERSION)" != "$(PYTHON_VERSION_REQUIRED)" ]; then \
		echo "Error: Python $(PYTHON_VERSION_REQUIRED) is prefered, but $(PYTHON_VERSION) is installed."; \
	fi

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
run_super_resolution: unet_model.py super_resolution_problem.py
	$(VENV_BIN)/$(PYTHON) super_resolution_problem.py

run_image_inpainting: unet_model.py image_inpainting_problem.py
	$(VENV_BIN)/$(PYTHON) image_inpainting_problem.py

.PHONY: install_venv venv install libs_install export_requirements download_dataset run_super_resolution

help:
	@echo "Usage: make <target> [OPTIONS]"
	@echo ""
	@echo "Description: This Makefile provides targets for running projects"
	@echo ""
	@echo "Available Targets:"
	@echo "venv:"
	@echo "		Description: Create .venv for environment"
	@echo "		Usage: make venv"
