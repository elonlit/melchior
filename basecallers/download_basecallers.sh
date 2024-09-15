#!/usr/bin/env bash
# Run this script inside the basecallers/ directory

# Download the latest version of the basecallers from the ONT website
# and extract them to the specified directory if it does not already exist
if [ ! -f dorado-0.7.3-linux-x64.tar.gz ]; then
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.7.3-linux-x64.tar.gz
fi

# Extract the basecaller if it does not already exist
if [ ! -d dorado-0.7.3-linux-x64 ]; then
    tar -xzvf dorado-0.7.3-linux-x64.tar.gz
fi

echo "Basecallers downloaded and extracted successfully."
echo "Downloading rna002_70bps_hac@v3..."

./dorado-0.7.3-linux-x64/bin/dorado download --model rna002_70bps_hac@v3