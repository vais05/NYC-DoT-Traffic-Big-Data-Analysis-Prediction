#!/bin/bash

# Define directories
NOTEBOOKS_DIR="./notebooks"
OUTPUT_DIR="./html_output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Install required packages if not already installed
pip install -r requirements.txt

# Run the notebook converter
python notebook_to_html_converter.py $NOTEBOOKS_DIR $OUTPUT_DIR

echo "HTML files generated in $OUTPUT_DIR"
echo "Open $OUTPUT_DIR/index.html to view the dashboard"