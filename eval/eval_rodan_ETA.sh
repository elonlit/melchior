#!/usr/bin/env bash

# Function to display time in a human-readable format
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local remaining_seconds=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $remaining_seconds
}

# Function to run basecalling with ETA
run_basecalling() {
    local dataset=$1
    local output_file=$2
    local start_time=$(date +%s)
    local total_files=$(find data/test/${dataset} -name "*.fast5" | wc -l)
    local processed_files=0

    python -m eval.basecall data/test/${dataset} -m rodan -b 1024 | 
    while IFS= read -r line
    do
        if [[ $line == ">"* ]]; then
            ((processed_files++))
            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time))
            if [ $processed_files -eq 0 ]; then
                eta="Calculating..."
            else
                files_per_second=$(echo "scale=2; $processed_files / $elapsed_time" | bc)
                remaining_files=$((total_files - processed_files))
                remaining_seconds=$(echo "scale=0; $remaining_files / $files_per_second" | bc)
                eta=$(format_time $remaining_seconds)
            fi
            progress=$((processed_files * 100 / total_files))
            echo -ne "${dataset}: Progress: $progress% ($processed_files/$total_files) | Elapsed: $(format_time $elapsed_time) | ETA: $eta\r"
        fi
        echo "$line" >> $output_file
    done
    echo -e "\n${dataset} basecalling completed."
}

# Make sure this only runs in the melchior/ directory of the project
if [ ! -f README.md ]; then
    echo "This script should only be run in the melchior/ directory of the project."
    exit 1
fi

echo "Benchmarking Rodan..."

mkdir -p eval/rodan_outputs
cd eval/rodan_outputs
mkdir -p arabidopsis-dataset human-dataset mouse-dataset poplar-dataset yeast-dataset
cd ../..

# Run basecalling for each dataset
datasets=("arabidopsis-dataset" "human-dataset" "mouse-dataset" "poplar-dataset" "yeast-dataset")

for dataset in "${datasets[@]}"; do
    output_file="eval/rodan_outputs/${dataset}/output.fasta"
    if [ ! -f "$output_file" ]; then
        echo "Basecalling ${dataset}..."
        run_basecalling $dataset $output_file
    else
        echo "${dataset} already basecalled. Skipping."
    fi
done

echo "Basecalling process completed."

# Use minimap2 to align the basecalled sequences to the reference genomes
echo "Aligning basecalled sequences to reference genomes..."

if [ ! -f README.md ]; then
    echo "wrong dir."
    exit 1
fi

for dataset in "${datasets[@]}"; do
    reference="data/test/transcriptomes/${dataset%%-*}_reference.fasta"
    input="eval/rodan_outputs/${dataset}/output.fasta"
    output="eval/rodan_outputs/${dataset}/aligned.sam"
    if [ -f "$output" ]; then
        echo "${dataset} already aligned. Skipping."
        continue
    fi
    echo "Aligning ${dataset}..."
    minimap2 --secondary=no -ax map-ont -t 32 --cs $reference $input > $output
done

echo "Alignment process completed."

# Run the evaluation script to calculate the accuracy metrics
echo "Calculating accuracy metrics..."

for dataset in "${datasets[@]}"; do
    echo "For ${dataset}:"
    python -m eval.accuracy eval/rodan_outputs/${dataset}/aligned.sam data/test/transcriptomes/${dataset%%-*}_reference.fasta > eval/rodan_outputs/${dataset}/accuracy.txt
done

echo "Evaluation process completed."