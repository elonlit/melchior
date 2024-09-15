#!/usr/bin/env bash

# This script takes an argument "limited" or "full" to determine whether to erase the full evaluation or a limited version

if [ ! -f README.md ]; then
    echo "This script should only be run in the melchior/ directory of the project."
    exit 1
fi

# Make sure the user has provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <limited|full>"
    exit 1
fi

# Make sure the user has provided a valid argument
if [ "$1" != "limited" ] && [ "$1" != "full" ]; then
    echo "Usage: $0 <limited|full>"
    exit 1
fi

# Remove the output files for the specified evaluation
if [ "$1" == "limited" ]; then
    rm -rf eval/sanity_check_outputs
    echo "Sanity check outputs removed."
else
    rm -rf eval/melchior_outputs
    rm -rf eval/rodan_outputs
    rm -rf eval/dorado_outputs
    echo "Full evaluation files removed."
fi