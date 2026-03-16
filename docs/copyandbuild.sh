#!/bin/bash

# Copy notebooks
echo "Copying notebooks..."

# Set directories
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${BASEDIR}/../application/SSCValidationHeligoland/"
DST="${BASEDIR}/source/notebooks/_temp/"

# Check if source directory exists
if [ ! -d "$SRC" ]; then
  echo "ERROR: Source folder does not exist: $SRC"
  echo "Current working directory: $(pwd)"
  exit 1
fi

# Copy all .ipynb files recursively
echo "Copying .ipynb files..."
find "$SRC" -type f -name "*.ipynb" -exec cp -v {} "$DST" \;

echo "Done copying notebooks."

# Generate API documentation
echo "Generating API docs..."
sphinx-apidoc -o source/api -f ../lidalign

# Clean and rebuild HTML
echo "Cleaning and rebuilding HTML..."
make clean
make html

# Remove copied notebooks
echo "Removing copied notebooks..."

# Remove all .ipynb files in _temp
echo "Removing .ipynb files from _temp..."
find "$DST" -type f -name "*.ipynb" -delete

echo "All done."