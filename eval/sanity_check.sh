#!/usr/bin/env bash
# This script runs a sanity check of the basecalling processs
# using a subset of the human transcriptome and the Melchior model

# Run the basecalling process a subset of the human transcriptome
if [ ! -f README.md ]; then
    echo "This script should only be run in the root directory of the project."
    exit 1
fi

echo "Basecalling sanity check dataset..."
mkdir -p eval/sanity_check_outputs
python -m eval.basecall data/sanity_check/test-data -m melchior -o eval/sanity_check_outputs/times.txt > eval/sanity_check_outputs/output.fasta

echo "Basecalling process completed."

# Use minimap2 to align the basecalled sequences to the reference genome
echo "Aligning basecalled sequences to the reference genome..."
minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/human_reference.fasta eval/sanity_check_outputs/output.fasta > eval/sanity_check_outputs/aligned.sam

echo "Alignment process completed."

# Run the evaluation script to calculate the accuracy metrics
echo "Running evaluation script..."
python -m eval.accuracy eval/sanity_check_outputs/aligned.sam data/test/transcriptomes/human_reference.fasta > eval/sanity_check_outputs/accuracy.txt