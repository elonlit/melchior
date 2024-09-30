#!/usr/bin/env bash

# Basecalling with Guppy
guppy_basecaller -i data/test/arabidopsis-dataset/ -s eval/guppy_outputs/arabidopsis-dataset/ -c dna_r9.4.1_450bps_hac.cfg --device cuda:0,1
guppy_basecaller -i data/test/human-dataset/ -s eval/guppy_outputs/human-dataset/ -c dna_r9.4.1_450bps_hac.cfg --device cuda:0,1
guppy_basecaller -i data/test/mouse-dataset/ -s eval/guppy_outputs/mouse-dataset/ -c dna_r9.4.1_450bps_hac.cfg --device cuda:0,1
guppy_basecaller -i data/test/poplar-dataset/ -s eval/guppy_outputs/poplar-dataset/ -c dna_r9.4.1_450bps_hac.cfg --device cuda:0,1
guppy_basecaller -i data/test/yeast-dataset/ -s eval/guppy_outputs/yeast-dataset/ -c dna_r9.4.1_450bps_hac.cfg --device cuda:0,1

# Check if the output.fasta file already exists for each dataset
if [ ! -f eval/guppy_outputs/arabidopsis-dataset/output.fasta ]; then
    echo "Basecalling Arabidopsis dataset..."
    cat eval/guppy_outputs/arabidopsis-dataset/workspace/pass/*.fastq > eval/guppy_outputs/arabidopsis-dataset/output.fastq
    seqtk seq -A eval/guppy_outputs/arabidopsis-dataset/output.fastq > eval/guppy_outputs/arabidopsis-dataset/output.fasta
fi


