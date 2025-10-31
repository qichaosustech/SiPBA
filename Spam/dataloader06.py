import os
import pandas as pd

# Set file paths
index_file = r'SPAMDATA/trec06p/full/index'  # Path to the index file
data_dir = r'SPAMDATA/trec06p/data'  # Path to the data folder
output_file = 'trec06p_emails.csv'

# Create a list to store email content and labels
data = []

# Read the index file and consolidate the data
with open(index_file, 'r', encoding='utf-8') as f:
    for line in f:
        label, file_path = line.strip().split()
        file_path = os.path.join(data_dir, file_path)  # Build the full email file path

        # Read the email content
        if os.path.exists(file_path):  # Ensure the file exists
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as mail_file:
                message = mail_file.read()
                data.append((label, message))
        else:
            print(f"File {file_path} does not exist, skipping...")

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['label', 'message'])

# Save the data as a CSV file
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Data has been saved to {output_file}")
