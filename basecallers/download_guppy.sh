#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-g|-c]"
    echo "  -g: Download GPU version"
    echo "  -c: Download CPU version"
    exit 1
}

if [ "$#" -ne 1 ]; then
    usage
fi

GUPPY_URL="https://cdn.oxfordnanoportal.com/software/analysis/"

while getopts "gc" opt; do
    case ${opt} in
        g )
            VERSION="gpu"
            FILENAME="ont-guppy_6.5.7_linux64.tar.gz"
            ;;
        c )
            VERSION="cpu"
            FILENAME="ont-guppy-cpu_6.5.7_linux64.tar.gz"
            ;;
        \? )
            usage
            ;;
    esac
done

FULL_URL="${GUPPY_URL}${FILENAME}"

echo "Downloading $VERSION version of Guppy..."

# Download the file
wget --content-disposition "$FULL_URL"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi

# Extract the basecaller if it does not already exist
if [ ! -d ont_guppy ]; then
    tar -xzvf ont-guppy*.tar.gz
fi