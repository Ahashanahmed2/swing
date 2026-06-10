#!/usr/bin/env python3
import os

# Define the target directory
target_dir = "./output/ai_signal"

# List of files to check and delete
files = [
    "support_resistant.csv",
    "short_buy.csv",
    "bullish_strong.csv",
    "fail_short_buy_pass.csv",
    "rsi_diver.csv",
    "uptrand_buy.csv"
]

# Check if directory exists
if not os.path.isdir(target_dir):
    print(f"Error: Directory {target_dir} does not exist.")
    exit(1)

# Loop through files and delete if they exist
deleted_count = 0
for file in files:
    file_path = os.path.join(target_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        deleted_count += 1
    else:
        print(f"Not found: {file_path}")

# Summary
print("=================================")
print(f"Operation completed. {deleted_count} file(s) deleted.")
