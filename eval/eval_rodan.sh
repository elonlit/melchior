#!/usr/bin/env bash

# Make sure this only runs in the melchior/ directory of the project
if [ ! -f README.md ]; then
    echo "This script should only be run in the melchior/ directory of the project."
    exit 1
fi

echo "Benchmarking Rodan..."

mkdir -p eval/rodan_outputs
cd eval/rodan_outputs
mkdir -p arabidopsis-dataset
mkdir -p human-dataset
mkdir -p mouse-dataset
mkdir -p poplar-dataset
mkdir -p yeast-dataset
cd ../..

# Check if the output.fasta file already exists for each dataset
if [ ! -f eval/rodan_outputs/arabidopsis-dataset/output.fasta ]; then
    echo "Basecalling Arabidopsis dataset..."
    python -m eval.basecall data/test/arabidopsis-dataset -m rodan -b 128 > eval/rodan_outputs/arabidopsis-dataset/output.fasta
fi

if [ ! -f eval/rodan_outputs/human-dataset/output.fasta ]; then
    echo "Basecalling Human dataset..."
    python -m eval.basecall data/test/human-dataset -m rodan -b 128 > eval/rodan_outputs/human-dataset/output.fasta
fi

if [ ! -f eval/rodan_outputs/mouse-dataset/output.fasta ]; then
    echo "Basecalling Mouse dataset..."
    python -m eval.basecall data/test/mouse-dataset -m rodan -b 128 > eval/rodan_outputs/mouse-dataset/output.fasta
fi

if [ ! -f eval/rodan_outputs/poplar-dataset/output.fasta ]; then
    echo "Basecalling Poplar dataset..."
    python -m eval.basecall data/test/poplar-dataset -m rodan -b 128 > eval/rodan_outputs/poplar-dataset/output.fasta
fi

if [ ! -f eval/rodan_outputs/yeast-dataset/output.fasta ]; then
    echo "Basecalling Yeast dataset..."
    python -m eval.basecall data/test/yeast-dataset -m rodan -b 128 > eval/rodan_outputs/yeast-dataset/output.fasta
fi

echo "Basecalling process completed."

# Use minimap2 to align the basecalled sequences to the reference genomes
echo "Aligning basecalled sequences to reference genomes..."

if [ ! -f README.md ]; then
    echo "wrong dir."
    exit 1
fi

minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/arabidopsis_reference.fasta eval/rodan_outputs/arabidopsis-dataset/output.fasta > eval/rodan_outputs/arabidopsis-dataset/aligned.sam
minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/human_reference.fasta eval/rodan_outputs/human-dataset/output.fasta > eval/rodan_outputs/human-dataset/aligned.sam
minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/mouse_reference.fasta eval/rodan_outputs/mouse-dataset/output.fasta > eval/rodan_outputs/mouse-dataset/aligned.sam
minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/poplar_reference.fasta eval/rodan_outputs/poplar-dataset/output.fasta > eval/rodan_outputs/poplar-dataset/aligned.sam
minimap2 --secondary=no -ax map-ont -t 32 --cs data/test/transcriptomes/yeast_reference.fasta eval/rodan_outputs/yeast-dataset/output.fasta > eval/rodan_outputs/yeast-dataset/aligned.sam

echo "Alignment process completed."

# Run the evaluation script to calculate the accuracy metrics
echo "Calculating accuracy metrics..."

cd ..
echo "For Arabidopsis dataset:"
python -m eval.accuracy eval/rodan_outputs/arabidopsis-dataset/aligned.sam data/test/transcriptomes/arabidopsis_reference.fasta > eval/rodan_outputs/arabidopsis-dataset/accuracy.txt

echo "For Human dataset:"
python -m eval.accuracy eval/rodan_outputs/human-dataset/aligned.sam data/test/transcriptomes/human_reference.fasta > eval/rodan_outputs/human-dataset/accuracy.txt

echo "For Mouse dataset:"
python -m eval.accuracy eval/rodan_outputs/mouse-dataset/aligned.sam data/test/transcriptomes/mouse_reference.fasta > eval/rodan_outputs/mouse-dataset/accuracy.txt

echo "For Poplar dataset:"
python -m eval.accuracy eval/rodan_outputs/poplar-dataset/aligned.sam data/test/transcriptomes/poplar_reference.fasta > eval/rodan_outputs/poplar-dataset/accuracy.txt

echo "For Yeast dataset:"
python -m eval.accuracy eval/rodan_outputs/yeast-dataset/aligned.sam data/test/transcriptomes/yeast_reference.fasta > eval/rodan_outputs/yeast-dataset/accuracy.txt

echo "Evaluation process completed."
