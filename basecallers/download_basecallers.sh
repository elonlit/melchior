#!/usr/bin/env bash
# This script downloads all of the necessary basecallers for the benchmarking process
# Run this script inside the basecallers/ directory

./download_melchior.sh
./download_gcrtcall.sh -g
./download_guppy.sh