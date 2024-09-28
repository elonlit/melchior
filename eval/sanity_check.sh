#!/usr/bin/env bash
# This script runs a sanity check of the basecalling process
# using a subset of the human transcriptome and the specified model

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 {rodan, melchior, gcrtcall}"
    exit 1
fi

# Validate the argument
case $1 in
    rodan|melchior|gcrtcall)
        model=$1
        ;;
    *)
        echo "Usage: $0 {rodan, melchior, gcrtcall}"
        exit 1
        ;;
esac

# Run the basecalling process a subset of the human transcriptome
if [ ! -f README.md ]; then
    echo "This script should only be run in the root directory of the project."
    exit 1
fi

echo "Basecalling sanity check dataset using $model model..."
mkdir -p eval/sanity_check_outputs
python -m eval.basecall data/sanity_check/test-data -m $model -o eval/sanity_check_outputs/times.txt > eval/sanity_check_outputs/output.fasta

echo "Basecalling process completed."

# Use minimap2 to align the basecalled sequences to the reference genome
echo "Aligning basecalled sequences to the reference genome..."
minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/human_reference.fasta eval/sanity_check_outputs/output.fasta > eval/sanity_check_outputs/aligned.sam

echo "Alignment process completed."

# Run the evaluation script to calculate the accuracy metrics
echo "Running evaluation script..."
python -m eval.accuracy eval/sanity_check_outputs/aligned.sam data/test/transcriptomes/human_reference.fasta > eval/sanity_check_outputs/accuracy.txt