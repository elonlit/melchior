#!/usr/bin/env bash

# URLs for the files
url_train="https://zenodo.org/records/4556951/files/rna-train.hdf5?download=1"
url_valid="https://zenodo.org/records/4556951/files/rna-valid.hdf5?download=1"

# Function to download file if it doesn't exist
download_if_not_exists() {
    local url=$1
    local filename=$2

    if [ -f "$filename" ]; then
        echo "$filename already exists. Skipping download."
    else
        echo "Downloading $filename..."
        aria2c -x 16 -s 16 -o "$filename" "$url"
        if [ $? -eq 0 ]; then
            echo "$filename downloaded successfully."
        else
            echo "Failed to download $filename."
            rm -f "$filename"  # Remove the partial download if it failed
        fi
    fi
}

# Download rna-train.hdf5
download_if_not_exists "$url_train" "rna-train.hdf5"

# Download rna-valid.hdf5
download_if_not_exists "$url_valid" "rna-valid.hdf5"

echo "Download process completed."