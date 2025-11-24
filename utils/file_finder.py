# utils/file_finder.py
import os

def find_files(base_path, extensions):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                yield os.path.join(root, file)
