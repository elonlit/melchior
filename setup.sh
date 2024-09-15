#!/usr/bin/env bash

create_and_activate_melchior() {
    local ENV_NAME="melchior"

    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 could not be found. Please install Python 3 and try again."
        return 1
    fi

    # Check if Conda is available
    if ! command -v conda &> /dev/null; then
        echo "Conda could not be found. Please install Conda and try again."
        return 1
    fi

    # Create Conda environment from YAML file if it does not exist
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists."
    else
        echo "Creating Conda environment '$ENV_NAME'..."
        if conda env create -f environment.yml; then
            echo "Conda environment '$ENV_NAME' created successfully."
        else
            echo "Failed to create Conda environment '$ENV_NAME'. Exiting."
            return 1
        fi
    fi

    # Activate Conda environment
    echo "Activating Conda environment '$ENV_NAME'..."
    if conda activate "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' activated successfully."
        echo "Python version:"
        python --version
    else
        echo "Failed to activate Conda environment '$ENV_NAME'."
        return 1
    fi

    # Install additional pip dependencies
    if [ -f requirements.txt ]; then
        echo "Installing additional pip dependencies..."
        if pip install -r requirements.txt; then
            echo "Dependencies installed successfully."
        else
            echo "Failed to install dependencies."
            return 1
        fi
    else
        echo "No 'requirements.txt' file found. Skipping pip install."
    fi
}

# Check if the script is sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script should be sourced, not run directly."
    echo "Please use: source ${BASH_SOURCE[0]}"
else
    create_and_activate_melchior
fi
