import os
import pandas as pd

# Set the root directory path for the Enron_Spam dataset
data_dir = 'SPAMDATA/Enron_Spam'  # Path to your Enron Spam dataset

# Initialize lists to store email content and labels
emails = []
labels = []

# Iterate through each subfolder in the Enron_Spam folder (enron1, enron2, ...)
for subfolder in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir, subfolder)

    # Ensure to only process subfolders (enron1, enron2, ...)
    if os.path.isdir(subfolder_path):
        # Iterate through each subfolder's ham and spam folders
        for label in ['ham', 'spam']:
            label_folder_path = os.path.join(subfolder_path, label)

            # Check if the folder exists
            if not os.path.exists(label_folder_path):
                print(f"Folder {label_folder_path} does not exist!")
                continue

            # Iterate through each email file in the folder (.txt)
            for file_name in os.listdir(label_folder_path):
                file_path = os.path.join(label_folder_path, file_name)

                try:
                    with open(file_path, 'r', encoding='latin1') as f:
                        content = f.read()  # Read the email content

                    # Add the email content and label to the lists
                    emails.append(content)
                    labels.append(label)  # 'spam' or 'ham' as the label

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}, skipping this email.")
                    continue

# Create a DataFrame to store email content and labels
df = pd.DataFrame({'label': labels, 'message': emails})
# Shuffle the dataset (randomly reorder)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Set the path to store the file
output_file = 'Enron_Spam_emails.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_file, index=False, encoding='utf-8')

# Print success message
print(f"Data has been saved to {os.path.abspath(output_file)}")
