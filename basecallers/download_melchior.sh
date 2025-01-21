#!/usr/bin/env bash
# Run this script inside the basecallers/ directory

if command -v wget &> /dev/null; then
    wget https://huggingface.co/elonlit/Melchior/resolve/main/melchior.pth
elif command -v curl &> /dev/null; then
    curl -L -O https://huggingface.co/elonlit/Melchior/resolve/main/melchior.pth
else
    echo "Error: Neither wget nor curl is installed"
    exit 1
fi