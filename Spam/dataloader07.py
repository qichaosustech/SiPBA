import os
import pandas as pd

# Set file paths
index_file = 'SPAMDATA/trec07p/full/index'  # Path to the index file
data_dir = 'SPAMDATA/trec07p/data'  # Path to the data folder

# Initialize lists to store email content and labels
emails = []
labels = []

# Read the index file
with open(index_file, 'r', encoding='latin1') as f:
    for line in f:
        # Split each line to get the label and email file path
        label, file_path = line.strip().split()
        file_name = file_path.replace('../data/', '')  # Get the email file name

        # Build the full path to the email file
        mail_file_path = os.path.join(data_dir, file_name)

        # Use try-except to skip files that can't be read
        try:
            with open(mail_file_path, 'r', encoding='latin1') as mail_file:
                content = mail_file.read()

            # Add the email content and label to the lists
            emails.append(content)
            labels.append(label)

        except FileNotFoundError:
            print(f"File {mail_file_path} not found, skipping this email.")
            continue  # Skip this email

        except Exception as e:
            print(f"Error reading file {mail_file_path}: {e}, skipping this email.")
            continue  # Skip this email

# Create a DataFrame to store email content and labels
df = pd.DataFrame({'label': labels, 'message': emails})

# Set the path to save the file
output_file = 'trec07p_emails.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False, encoding='utf-8')

# Print success message
print(f"Data has been saved to {os.path.abspath(output_file)}")
