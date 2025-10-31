import os
import pandas as pd

# Set the root directory for the Ling-Spam dataset (modify according to your actual directory structure)
corpus_root = r'./SPAMDATA/lingspam_public/lingspam_public'  # Assume this directory contains "bare", "lemm", "lemm_stop", "stop" subdirectories

# Choose a version folder, e.g., "lemm_stop"
version_folder = 'lemm_stop'
version_path = os.path.join(corpus_root, version_folder)

# Create a list to store email content and labels
data = []

# Iterate through 10 partition directories (part1, part2, ..., part10)
for part in os.listdir(version_path):
    part_path = os.path.join(version_path, part)
    if os.path.isdir(part_path):
        for filename in os.listdir(part_path):
            file_path = os.path.join(part_path, filename)
            if os.path.isfile(file_path):
                # Determine the label: files starting with "spmsg" are labeled as spam, others as legitimate
                if filename.startswith("spmsg"):
                    label = 'spam'
                else:
                    label = 'ham'

                # Read the email content, some files may use latin-1 encoding
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    message = f.read()
                data.append((label, message))

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['label', 'message'])

# Save as a CSV file
output_file = f'LingSpam_emails.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Data has been saved to {output_file}")
