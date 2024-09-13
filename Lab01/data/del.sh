#!/bin/bash

# Define the pattern for files to delete
pattern1="pthread*"
pattern2="tbb*"
pattern3="serial*"

# Use the rm command with the pattern to delete matching files
rm -f $pattern1 $pattern2 $pattern3

echo "Deleted all image files starting with 'pthread'"
