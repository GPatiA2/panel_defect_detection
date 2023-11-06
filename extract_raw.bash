#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <X_directory> <Y_directory>"
  exit 1
fi

X="$1"
Y="$2"

if [ ! -d "$X" ]; then
  echo "Directory $X does not exist."
  exit 1
fi

if [ ! -d "$Y" ]; then
  echo "Directory $Y does not exist."
  exit 1
fi

# Check if dji_irp is installed and in the PATH
if ! command -v dji_irp &>/dev/null; then
  echo "dji_irp command not found. Please make sure it's installed and in your PATH."
  exit 1
fi

# Iterate over JPG files in directory X
for jpg_file in "$X"/*.JPG; do
  if [ -f "$jpg_file" ]; then
    # Extract the file name (without the path) from the full file path
    file_name=$(basename "$jpg_file")

    # Replace ".JPEG" with ".raw" in the output file name
    output_file="$Y/$(basename "$file_name").raw"

    # Execute the dji_irp command
    dji_irp -s "$X/$file_name" -o "$output_file" --measurefmt float32 -a measure -v 1

    # Optionally, you can add error handling or logging here if needed
  fi
done

