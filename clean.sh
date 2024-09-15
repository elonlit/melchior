#!/usr/bin/env bash

cleanup_conda_env() {
    local ENV_NAME="melchior"

    # Function to check if the environment exists
    environment_exists() {
        conda env list | grep -q "$ENV_NAME"
    }

    # Check if Conda is available
    if ! command -v conda &> /dev/null; then
        echo "Error: Conda could not be found. Please install Conda and try again."
        return 1
    fi

    # Deactivate any active Conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "Deactivating current Conda environment: $CONDA_DEFAULT_ENV"
        conda deactivate
    fi

    if environment_exists; then
        echo "Conda environment '$ENV_NAME' found. Proceeding with deletion..."

        echo "Removing Conda environment '$ENV_NAME'..."
        conda env remove --name "$ENV_NAME" --all --yes

        if environment_exists; then
            echo "Error: Failed to remove Conda environment '$ENV_NAME'. Please check for any running processes using this environment and try again."
            return 1
        else
            echo "Conda environment '$ENV_NAME' has been successfully removed."
        fi
    else
        echo "Conda environment '$ENV_NAME' does not exist. No action needed."
    fi

    # Clean Conda caches
    echo "Cleaning Conda caches..."
    conda clean --all --yes

    echo "Clean-up process completed."
}

# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script should be sourced, not executed directly."
    echo "Please use: source ${BASH_SOURCE[0]}"
else
    cleanup_conda_env
fi
