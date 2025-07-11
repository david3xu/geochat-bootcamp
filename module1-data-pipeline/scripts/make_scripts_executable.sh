#!/bin/bash
# Make all Python scripts in the scripts directory executable

# Change to the scripts directory
cd "$(dirname "$0")" || exit 1

# Make all Python scripts executable
chmod +x *.py

echo "All scripts are now executable." 