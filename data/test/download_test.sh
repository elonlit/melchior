#!/usr/bin/env bash

# URLs for the files
arabidopsis="https://zenodo.org/records/4557005/files/arabidopsis-dataset.tgz?download=1"
human="https://zenodo.org/records/4557005/files/human-dataset.tgz?download=1"
mouse="https://zenodo.org/records/4557005/files/mouse-dataset.tgz?download=1"
poplar="https://zenodo.org/records/4557005/files/poplar-dataset.tgz?download=1"
transcriptomes="https://zenodo.org/records/4557005/files/transcriptomes.tgz?download=1"
yeast="https://zenodo.org/records/4557005/files/yeast-dataset.tgz?download=1"

# Function to download file if it doesn't exist
download_if_not_exists() {
    local url=$1
    local filename=$2

    if [ -f "$filename" ]; then
        echo "$filename already exists. Skipping download."
    else
        echo "Downloading $filename..."
        wget -O "$filename" "$url"
        if [ $? -eq 0 ]; then
            echo "$filename downloaded successfully."
        else
            echo "Failed to download $filename."
            rm -f "$filename"  # Remove the partial download if it failed
        fi
    fi
}

# Download .tgz files
download_if_not_exists "$arabidopsis" "arabidopsis-dataset.tgz"
download_if_not_exists "$human" "human-dataset.tgz"
download_if_not_exists "$mouse" "mouse-dataset.tgz"
download_if_not_exists "$poplar" "poplar-dataset.tgz"
download_if_not_exists "$transcriptomes" "transcriptomes.tgz"
download_if_not_exists "$yeast" "yeast-dataset.tgz"

echo "Download process completed."

# Function to extract and rename the tgz files
extract_and_rename() {
    local filename=$1
    local dir_name=$2

    if [ ! -d "$dir_name" ]; then
        echo "Extracting $filename..."
        tar -xzf "$filename"
        mv dataset "$dir_name" # Rename the extracted directory
    fi
}

# Extract the .tgz files if they haven't already been extracted
echo "Extracting .tgz files if necessary..."

extract_and_rename "arabidopsis-dataset.tgz" "arabidopsis-dataset"
extract_and_rename "human-dataset.tgz" "human-dataset"
extract_and_rename "mouse-dataset.tgz" "mouse-dataset"
extract_and_rename "poplar-dataset.tgz" "poplar-dataset"
extract_and_rename "yeast-dataset.tgz" "yeast-dataset"
tar -xzf "transcriptomes.tgz" # Extract the transcriptomes.tgz file
mkdir transcriptomes
# Move all .fasta files to the transcriptomes directory
mv *.fasta transcriptomes