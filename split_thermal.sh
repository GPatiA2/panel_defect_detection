#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 source_dir destination_dir"
  exit 1
fi

# Source and destination directories
source_dir="$1"
dest_dir="$2"

# Verify that source_dir exists
if [ ! -d "$source_dir" ]; then
  echo "Source directory does not exist: $source_dir"
  exit 1
fi

# Verify that destination_dir exists or create it
if [ ! -d "$dest_dir" ]; then
  mkdir -p "$dest_dir"
fi

# Copy files with names ending in "_T.JPG" to the destination directory
for file in "$source_dir"/*_T.JPG; do
  if [ -f "$file" ]; then
    cp "$file" "$dest_dir"
    echo "Copied: $file"
  fi
done

echo "Copying completed."

