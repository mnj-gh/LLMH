# This script will:

# Read all .txt files in a folder

# Merge broken lines

# Remove metadata ((Berliner...Zeit) etc.)

# Flatten to clean sentences/paragraphs

# Output a file like homeo_corpus.txt for training.

import os
import re

# === Change this to your folder containing raw .txt files ===
input_folder = r"medicines_data"
output_file = "homeo_corpus.txt"

corpus = []

def clean_line(line):
    # Remove excessive whitespace
    line = line.strip()

    # Remove metadata references inside brackets
    line = re.sub(r'\([^)]*?(berliner|zeit|vol|hom|arzte)[^)]*?\)', '', line, flags=re.IGNORECASE)

    # Remove lines that start with citations or metadata
    if re.match(r'^\(?.*?(berliner|zeit|vol|arzte).*?\)?', line, re.IGNORECASE):
        return ""

    # Fix common separators
    line = re.sub(r'\s*--\s*', ': ', line)
    line = re.sub(r'\s{2,}', ' ', line)  # collapse multiple spaces

    return line

# Helper to check if line looks like it ends naturally
def ends_cleanly(line):
    return line.endswith(('.', '?', '!'))

# Read all .txt files in folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            buffer = ""
            for line in f:
                line = clean_line(line)

                if not line:
                    continue  # skip empty or bad lines

                # Merge broken lines into one
                if buffer:
                    buffer += " " + line
                    if ends_cleanly(line):
                        corpus.append(buffer.strip())
                        buffer = ""
                else:
                    buffer = line
                    if ends_cleanly(line):
                        corpus.append(buffer.strip())
                        buffer = ""

            # Flush buffer if anything remains
            if buffer:
                corpus.append(buffer.strip())

# Write output corpus
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(corpus))

print(f"✅ Processed {len(corpus)} lines. Saved to: {output_file}")
