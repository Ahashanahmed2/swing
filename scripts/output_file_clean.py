#!/bin/bash

# Define the target directory
TARGET_DIR="./output/ai_signal"

# List of files to check and delete
FILES=(
    "support_resistant.csv"
    "short_buy.csv"
    "bullish_strong.csv"
    "fail_short_buy_pass.csv"
    "rsi_diver.csv"
)

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

# Loop through files and delete if they exist
deleted_count=0
for file in "${FILES[@]}"; do
    file_path="$TARGET_DIR/$file"
    if [ -f "$file_path" ]; then
        rm "$file_path"
        echo "Deleted: $file_path"
        ((deleted_count++))
    else
        echo "Not found: $file_path"
    fi
done

# Summary
echo "================================="
echo "Operation completed. $deleted_count file(s) deleted."
