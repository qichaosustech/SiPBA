from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import os
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
import pickle
import numpy as np
import argparse
import sys
import logging
from datetime import datetime
import os

import time

time.sleep(20000)
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_filename = datetime.now().strftime(f"{log_dir}/run_log_%Y%m%d_%H%M%S.txt")
log_file = open(log_filename, "a")
sys.stdout = log_file
sys.stderr = log_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding="utf-8"),
        logging.StreamHandler()
    ]
)


logging.info("Starting the script...")

def generate_data(dataset, vectorizer='None', train_size=0, filename='None', seed=1):
    # Define the directory to save the data
    save_directory = '/data/CQC/result'

    # Check if the directory exists, create it if not
    if not os.path.exists(f'{save_directory}/{filename}'):
        os.makedirs(f'{save_directory}/{filename}')

    # Check if the file already exists
    if not os.path.exists(f'{save_directory}/{filename}/{dataset}_X_test_seed{seed}.pth'):
        df = pd.read_csv(f'{dataset}_emails.csv')

        # Remove any rows with missing or blank 'message' values
        df = df[df['message'].str.strip().notna()]

        if train_size > 0:
            # Sample the training data
            train_df = df.sample(n=train_size, random_state=seed)
            remaining_df = df.drop(train_df.index)
            train_df = train_df[train_df['message'].str.strip().notna()]
            remaining_df = remaining_df[remaining_df['message'].str.strip().notna()]

            # Vectorization
            vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_features=3000)
            X_train = vectorizer.fit_transform(train_df['message']).toarray()
            y_train = train_df['label'].apply(lambda x: 1 if x == 'ham' else -1).values  # 1 for ham, -1 for spam

            # Save the training and testing data
            torch.save(X_train, f'{save_directory}/{filename}/{dataset}_X_train_seed{seed}.pth')
            torch.save(y_train, f'{save_directory}/{filename}/{dataset}_y_train_seed{seed}.pth')

            X_test = vectorizer.transform(remaining_df['message']).toarray()
            y_test = remaining_df['label'].apply(lambda x: 1 if x == 'ham' else -1).values
            torch.save(X_test, f'{save_directory}/{filename}/{dataset}_X_test_seed{seed}.pth')
            torch.save(y_test, f'{save_directory}/{filename}/{dataset}_y_test_seed{seed}.pth')

            # Save the vectorizer
            with open(f'{save_directory}/{filename}/{dataset}_vectorizer_seed{seed}.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)

            return X_train, y_train, X_test, y_test, vectorizer
        else:
            X_test = vectorizer.transform(df['message']).toarray()
            y_test = df['label'].apply(lambda x: 1 if x == 'ham' else -1).values
            torch.save(X_test, f'{save_directory}/{filename}/{dataset}_X_test_seed{seed}.pth')
            torch.save(y_test, f'{save_directory}/{filename}/{dataset}_y_test_seed{seed}.pth')
            return X_test, y_test
    else:
        if train_size > 0:
            X_train = torch.load(f'{save_directory}/{filename}/{dataset}_X_train_seed{seed}.pth')
            y_train = torch.load(f'{save_directory}/{filename}/{dataset}_y_train_seed{seed}.pth')
            X_test = torch.load(f'{save_directory}/{filename}/{dataset}_X_test_seed{seed}.pth')
            y_test = torch.load(f'{save_directory}/{filename}/{dataset}_y_test_seed{seed}.pth')
            with open(f'{save_directory}/{filename}/{dataset}_vectorizer_seed{seed}.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            return X_train, y_train, X_test, y_test, vectorizer
        else:
            X_test = torch.load(f'{save_directory}/{filename}/{dataset}_X_test_seed{seed}.pth')
            y_test = torch.load(f'{save_directory}/{filename}/{dataset}_y_test_seed{seed}.pth')
            return X_test, y_test


def eval_model(X_test, y_test, w):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
    y_pred = torch.matmul(X_test_tensor, w)
    y_pred_label = torch.where(y_pred > 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
    f1 = f1_score(y_test_tensor.cpu().numpy(), y_pred_label.cpu().numpy())
    accuracy = (y_pred_label == y_test_tensor).sum().item() / len(y_test_tensor)
    return f1, accuracy




torch.random.manual_seed(123)
np.random.seed(123)
parser = argparse.ArgumentParser(description='Spam classifier training')
parser.add_argument('--dataset', type=str, default='LingSpam', metavar='N',
                    help='trec06p, trec07p, LingSpam, Enron_Spam')



datasets = ['trec07p','trec06p',  'LingSpam', 'Enron_Spam']
args = parser.parse_args()


for args.dataset in ['trec07p','trec06p',  'LingSpam', 'Enron_Spam']:
    PBO_hinge_trec06_f1_list = []
    PBO_hinge_trec06_accuracy_list = []
    PBO_hinge_trec07_f1_list = []
    PBO_hinge_trec07_accuracy_list = []
    PBO_hinge_LingSpam_f1_list = []
    PBO_hinge_LingSpam_accuracy_list = []
    PBO_hinge_Enron_Spam_f1_list = []
    PBO_hinge_Enron_Spam_accuracy_list = []

    PBO_crossentropy_trec06_f1_list = []
    PBO_crossentropy_trec06_accuracy_list = []
    PBO_crossentropy_trec07_f1_list = []
    PBO_crossentropy_trec07_accuracy_list = []
    PBO_crossentropy_LingSpam_f1_list = []
    PBO_crossentropy_LingSpam_accuracy_list = []
    PBO_crossentropy_Enron_Spam_f1_list = []
    PBO_crossentropy_Enron_Spam_accuracy_list = []

    SQP_hinge_trec06_f1_list = []
    SQP_hinge_trec06_accuracy_list = []
    SQP_hinge_trec07_f1_list = []
    SQP_hinge_trec07_accuracy_list = []
    SQP_hinge_LingSpam_f1_list = []
    SQP_hinge_LingSpam_accuracy_list = []
    SQP_hinge_Enron_Spam_f1_list = []
    SQP_hinge_Enron_Spam_accuracy_list = []

    SQP_CE_trec06_f1_list = []
    SQP_CE_trec06_accuracy_list = []
    SQP_CE_trec07_f1_list = []
    SQP_CE_trec07_accuracy_list = []
    SQP_CE_LingSpam_f1_list = []
    SQP_CE_LingSpam_accuracy_list = []
    SQP_CE_Enron_Spam_f1_list = []
    SQP_CE_Enron_Spam_accuracy_list = []

    svm_trec06_f1_list = []
    svm_trec06_accuracy_list = []
    svm_trec07_f1_list = []
    svm_trec07_accuracy_list = []
    svm_LingSpam_f1_list = []
    svm_LingSpam_accuracy_list = []
    svm_Enron_Spam_f1_list = []
    svm_Enron_Spam_accuracy_list = []

    logreg_trec06_f1_list = []
    logreg_trec06_accuracy_list = []
    logreg_trec07_f1_list = []
    logreg_trec07_accuracy_list = []
    logreg_LingSpam_f1_list = []
    logreg_LingSpam_accuracy_list = []
    logreg_Enron_Spam_f1_list = []
    logreg_Enron_Spam_accuracy_list = []

    # Calculate and display averages for each list
    def display_average_scores():

        # Display averages for SVM
        print("\nSVM Averages:")
        print(f"trec06 Accuracy: {round(sum(svm_trec06_accuracy_list) / len(svm_trec06_accuracy_list), 4)}, F1: {round(sum(svm_trec06_f1_list) / len(svm_trec06_f1_list), 4)}")
        print(f"trec07 Accuracy: {round(sum(svm_trec07_accuracy_list) / len(svm_trec07_accuracy_list), 4)}, F1: {round(sum(svm_trec07_f1_list) / len(svm_trec07_f1_list), 4)}")
        print(f"Enron_Spam Accuracy: {round(sum(svm_Enron_Spam_accuracy_list) / len(svm_Enron_Spam_accuracy_list), 4)}, F1: {round(sum(svm_Enron_Spam_f1_list) / len(svm_Enron_Spam_f1_list), 4)}")
        print(f"LingSpam Accuracy: {round(sum(svm_LingSpam_accuracy_list) / len(svm_LingSpam_accuracy_list), 4)}, F1: {round(sum(svm_LingSpam_f1_list) / len(svm_LingSpam_f1_list), 4)}")

        # Display averages for Logistic Regression
        print("\nLogistic Regression Averages:")
        print(f"trec06 Accuracy: {round(sum(logreg_trec06_accuracy_list) / len(logreg_trec06_accuracy_list), 4)}, F1: {round(sum(logreg_trec06_f1_list) / len(logreg_trec06_f1_list), 4)}")
        print(f"trec07 Accuracy: {round(sum(logreg_trec07_accuracy_list) / len(logreg_trec07_accuracy_list), 4)}, F1: {round(sum(logreg_trec07_f1_list) / len(logreg_trec07_f1_list), 4)}")
        print(f"Enron_Spam Accuracy: {round(sum(logreg_Enron_Spam_accuracy_list) / len(logreg_Enron_Spam_accuracy_list), 4)}, F1: {round(sum(logreg_Enron_Spam_f1_list) / len(logreg_Enron_Spam_f1_list), 4)}")
        print(f"LingSpam Accuracy: {round(sum(logreg_LingSpam_accuracy_list) / len(logreg_LingSpam_accuracy_list), 4)}, F1: {round(sum(logreg_LingSpam_f1_list) / len(logreg_LingSpam_f1_list), 4)}")
        # Display averages for SQP Hinge'''
        print("\nSQP Hinge Averages:")
        print(
            f"trec06 Accuracy: {round(sum(SQP_hinge_trec06_accuracy_list) / len(SQP_hinge_trec06_accuracy_list), 4)}, F1: {round(sum(SQP_hinge_trec06_f1_list) / len(SQP_hinge_trec06_f1_list), 4)}")
        print(
            f"trec07 Accuracy: {round(sum(SQP_hinge_trec07_accuracy_list) / len(SQP_hinge_trec07_accuracy_list), 4)}, F1: {round(sum(SQP_hinge_trec07_f1_list) / len(SQP_hinge_trec07_f1_list), 4)}")
        print(
            f"Enron_Spam Accuracy: {round(sum(SQP_hinge_Enron_Spam_accuracy_list) / len(SQP_hinge_Enron_Spam_accuracy_list), 4)}, F1: {round(sum(SQP_hinge_Enron_Spam_f1_list) / len(SQP_hinge_Enron_Spam_f1_list), 4)}")
        print(
            f"LingSpam Accuracy: {round(sum(SQP_hinge_LingSpam_accuracy_list) / len(SQP_hinge_LingSpam_accuracy_list), 4)}, F1: {round(sum(SQP_hinge_LingSpam_f1_list) / len(SQP_hinge_LingSpam_f1_list), 4)}")


        # Display averages for SQP CE
        print("\nSQP CE Averages:")
        print(f"trec06 Accuracy: {round(sum(SQP_CE_trec06_accuracy_list) / len(SQP_CE_trec06_accuracy_list), 4)}, F1: {round(sum(SQP_CE_trec06_f1_list) / len(SQP_CE_trec06_f1_list), 4)}")
        print(f"trec07 Accuracy: {round(sum(SQP_CE_trec07_accuracy_list) / len(SQP_CE_trec07_accuracy_list), 4)}, F1: {round(sum(SQP_CE_trec07_f1_list) / len(SQP_CE_trec07_f1_list), 4)}")
        print(f"Enron_Spam Accuracy: {round(sum(SQP_CE_Enron_Spam_accuracy_list) / len(SQP_CE_Enron_Spam_accuracy_list), 4)}, F1: {round(sum(SQP_CE_Enron_Spam_f1_list) / len(SQP_CE_Enron_Spam_f1_list), 4)}")
        print(f"LingSpam Accuracy: {round(sum(SQP_CE_LingSpam_accuracy_list) / len(SQP_CE_LingSpam_accuracy_list), 4)}, F1: {round(sum(SQP_CE_LingSpam_f1_list) / len(SQP_CE_LingSpam_f1_list), 4)}")

        
        # Display averages for PBO Hinge
        print("\nPBO Hinge Averages:")
        print(
            f"trec06 Accuracy: {round(sum(PBO_hinge_trec06_accuracy_list) / len(PBO_hinge_trec06_accuracy_list), 4)}, F1: {round(sum(PBO_hinge_trec06_f1_list) / len(PBO_hinge_trec06_f1_list), 4)}")
        print(
            f"trec07 Accuracy: {round(sum(PBO_hinge_trec07_accuracy_list) / len(PBO_hinge_trec07_accuracy_list), 4)}, F1: {round(sum(PBO_hinge_trec07_f1_list) / len(PBO_hinge_trec07_f1_list), 4)}")
        print(
            f"Enron_Spam Accuracy: {round(sum(PBO_hinge_Enron_Spam_accuracy_list) / len(PBO_hinge_Enron_Spam_accuracy_list), 4)}, F1: {round(sum(PBO_hinge_Enron_Spam_f1_list) / len(PBO_hinge_Enron_Spam_f1_list), 4)}")
        print(
            f"LingSpam Accuracy: {round(sum(PBO_hinge_LingSpam_accuracy_list) / len(PBO_hinge_LingSpam_accuracy_list), 4)}, F1: {round(sum(PBO_hinge_LingSpam_f1_list) / len(PBO_hinge_LingSpam_f1_list), 4)}")

        # Display averages for PBO CrossEntropy
        print("\nPBO CrossEntropy Averages:")
        print(
            f"trec06 Accuracy: {round(sum(PBO_crossentropy_trec06_accuracy_list) / len(PBO_crossentropy_trec06_accuracy_list), 4)}, F1: {round(sum(PBO_crossentropy_trec06_f1_list) / len(PBO_crossentropy_trec06_f1_list), 4)}")
        print(
            f"trec07 Accuracy: {round(sum(PBO_crossentropy_trec07_accuracy_list) / len(PBO_crossentropy_trec07_accuracy_list), 4)}, F1: {round(sum(PBO_crossentropy_trec07_f1_list) / len(PBO_crossentropy_trec07_f1_list), 4)}")
        print(
            f"Enron_Spam Accuracy: {round(sum(PBO_crossentropy_Enron_Spam_accuracy_list) / len(PBO_crossentropy_Enron_Spam_accuracy_list), 4)}, F1: {round(sum(PBO_crossentropy_Enron_Spam_f1_list) / len(PBO_crossentropy_Enron_Spam_f1_list), 4)}")
        print(
            f"LingSpam Accuracy: {round(sum(PBO_crossentropy_LingSpam_accuracy_list) / len(PBO_crossentropy_LingSpam_accuracy_list), 4)},  F1: {round(sum(PBO_crossentropy_LingSpam_f1_list) / len(PBO_crossentropy_LingSpam_f1_list), 4)}")

    for seed in range(10):
        train_data=args.dataset
        filename=f"trained_by_{train_data}"
        X_train, y_train, X_test, y_test,vectorizer=generate_data(train_data, train_size=500, filename= filename,seed=seed)

        globals()[f'X_{train_data}'] = X_test
        globals()[f'y_{train_data}'] = y_test
        for test_data in [dataset for dataset in datasets if dataset != train_data]:
            X, y = generate_data(test_data, vectorizer=vectorizer, filename=filename,seed=seed)
            globals()[f'X_{test_data}'] = X
            globals()[f'y_{test_data}'] = y
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


        w_SQP_hinge_loaded = torch.load(f'/data/CQC/result/{filename}/model_SQP_hinge_weights_seed{seed}.pth',
                                        weights_only=True)
        w_SQP_hinge_loaded = w_SQP_hinge_loaded.to(torch.float32)
        w_SQP_hinge = w_SQP_hinge_loaded.to(device)

        # Define datasets and corresponding result lists for PBO_CE
        datasets_SQP_hinge = [
        ("trec07", X_trec07p, y_trec07p, SQP_hinge_trec07_f1_list, SQP_hinge_trec07_accuracy_list),
        ("trec06", X_trec06p, y_trec06p, SQP_hinge_trec06_f1_list, SQP_hinge_trec06_accuracy_list),
        ("LingSpam", X_LingSpam, y_LingSpam, SQP_hinge_LingSpam_f1_list, SQP_hinge_LingSpam_accuracy_list),
        ("Enron_Spam", X_Enron_Spam, y_Enron_Spam, SQP_hinge_Enron_Spam_f1_list, SQP_hinge_Enron_Spam_accuracy_list),
        ]

        # Evaluate model on each dataset and collect metrics
        for name, X, y, f1_list, acc_list in datasets_SQP_hinge:
            f1, acc = eval_model(X, y, w_SQP_hinge)  # Evaluate with current dataset
            print(f"SQP_hinge in {name}: Accuracy: {acc}, f1_score: {f1}")
            f1_list.append(f1)  # Store F1 score
            acc_list.append(acc)  # Store accuracy

        w_SQP_CE_loaded = torch.load(f'/data/CQC/result/{filename}/model_SQP_CE_weights_seed{seed}.pth',
                                        weights_only=True)
        w_SQP_CE_loaded = w_SQP_CE_loaded.to(torch.float32)
        w_SQP_CE = w_SQP_CE_loaded.to(device)

        # Define datasets and corresponding result lists for PBO_CE
        datasets_SQP_CE = [
            ("trec07", X_trec07p, y_trec07p, SQP_CE_trec07_f1_list, SQP_CE_trec07_accuracy_list),
            ("trec06", X_trec06p, y_trec06p, SQP_CE_trec06_f1_list, SQP_CE_trec06_accuracy_list),
            ("LingSpam", X_LingSpam, y_LingSpam, SQP_CE_LingSpam_f1_list, SQP_CE_LingSpam_accuracy_list),
            ("Enron_Spam", X_Enron_Spam, y_Enron_Spam, SQP_CE_Enron_Spam_f1_list, SQP_CE_Enron_Spam_accuracy_list),
        ]

        # Evaluate model on each dataset and collect metrics
        for name, X, y, f1_list, acc_list in datasets_SQP_CE:
            f1, acc = eval_model(X, y, w_SQP_CE)  # Evaluate with current dataset
            print(f"SQP_CE in {name}: Accuracy: {acc}, f1_score: {f1}")
            f1_list.append(f1)  # Store F1 score
            acc_list.append(acc)  # Store accuracy



        # Load model weights and move to the specified device
        w_loaded = torch.load(f'/data/CQC/result/{filename}/model_weights_seed{seed}.pth', weights_only=True)
        w = w_loaded.to(device)

        # Define datasets and corresponding result lists
        datasets_PBO_hinge = [
            ("trec07", X_trec07p, y_trec07p, PBO_hinge_trec07_f1_list, PBO_hinge_trec07_accuracy_list),
            ("trec06", X_trec06p, y_trec06p, PBO_hinge_trec06_f1_list, PBO_hinge_trec06_accuracy_list),
            ("LingSpam", X_LingSpam, y_LingSpam, PBO_hinge_LingSpam_f1_list, PBO_hinge_LingSpam_accuracy_list),
            ("Enron_Spam", X_Enron_Spam, y_Enron_Spam, PBO_hinge_Enron_Spam_f1_list, PBO_hinge_Enron_Spam_accuracy_list),
        ]

        # Evaluate model on each dataset and collect metrics
        for name, X, y, f1_list, acc_list in datasets_PBO_hinge:
            f1, acc = eval_model(X, y, w)  # Evaluate the model
            print(f"PBO_hinge in {name}: Accuracy: {acc}, f1_score: {f1}")
            f1_list.append(f1)  # Store F1 score
            acc_list.append(acc)  # Store accuracy

        # Load cross-entropy model weights and move to device
        w_2_loaded = torch.load(f'/data/CQC/result/{filename}/model_2_weights_seed{seed}.pth', weights_only=True)
        w_2 = w_2_loaded.to(device)

        # Define datasets and corresponding result lists for PBO_CE
        datasets_PBO_CE = [
            ("trec07", X_trec07p, y_trec07p, PBO_crossentropy_trec07_f1_list, PBO_crossentropy_trec07_accuracy_list),
            ("trec06", X_trec06p, y_trec06p, PBO_crossentropy_trec06_f1_list, PBO_crossentropy_trec06_accuracy_list),
            (
            "LingSpam", X_LingSpam, y_LingSpam, PBO_crossentropy_LingSpam_f1_list, PBO_crossentropy_LingSpam_accuracy_list),
            ("Enron_Spam", X_Enron_Spam, y_Enron_Spam, PBO_crossentropy_Enron_Spam_f1_list,
                PBO_crossentropy_Enron_Spam_accuracy_list),
        ]

        # Evaluate model on each dataset and collect metrics
        for name, X, y, f1_list, acc_list in datasets_PBO_CE:
            f1, acc = eval_model(X, y, w_2)  # Evaluate with current dataset
            print(f"PBO_CE in {name}: Accuracy: {acc}, f1_score: {f1}")
            f1_list.append(f1)  # Store F1 score
            acc_list.append(acc)  # Store accuracy



        
        # Load trained SVM model
        with open(f'/data/CQC/result/{filename}/svm_model_seed{seed}.pkl', 'rb') as file:
            svm_model = pickle.load(file)
        # Define datasets and corresponding result lists for SVM
        datasets_svm = [
            ("trec07p", X_trec07p, y_trec07p, svm_trec07_f1_list, svm_trec07_accuracy_list),
            ("trec06p", X_trec06p, y_trec06p, svm_trec06_f1_list, svm_trec06_accuracy_list),
            ("LingSpam", X_LingSpam, y_LingSpam, svm_LingSpam_f1_list, svm_LingSpam_accuracy_list),
            ("Enron_Spam", X_Enron_Spam, y_Enron_Spam, svm_Enron_Spam_f1_list, svm_Enron_Spam_accuracy_list),
        ]
        # Predict and evaluate on each dataset
        for name, X, y, f1_list, acc_list in datasets_svm:
            y_pred = svm_model.predict(X)  # Make predictions
            acc = accuracy_score(y, y_pred)  # Calculate accuracy
            f1 = f1_score(y, y_pred)  # Calculate F1 score
            print(f"SVM, acc in {name}:{acc}, f1_score:{f1}")
            f1_list.append(f1)  # Record F1 score
            acc_list.append(acc)  # Record accuracy


        # Load the trained logistic regression model
        with open(f'/data/CQC/result/{filename}/logreg_model_seed{seed}.pkl', 'rb') as file:
            logreg = pickle.load(file)

        # Define datasets with their corresponding inputs, labels, and result lists
        datasets_logreg = [
            ("trec07p", X_trec07p, y_trec07p, logreg_trec07_f1_list, logreg_trec07_accuracy_list),
            ("trec06p", X_trec06p, y_trec06p, logreg_trec06_f1_list, logreg_trec06_accuracy_list),
            ("LingSpam", X_LingSpam, y_LingSpam, logreg_LingSpam_f1_list, logreg_LingSpam_accuracy_list),
            ("Enron_Spam", X_Enron_Spam, y_Enron_Spam, logreg_Enron_Spam_f1_list, logreg_Enron_Spam_accuracy_list),
        ]

        # Iterate over each dataset to evaluate the model and record metrics
        for name, X, y, f1_list, acc_list in datasets_logreg:
            y_pred = logreg.predict(X)  # Make predictions
            acc = accuracy_score(y, y_pred)  # Calculate accuracy
            f1 = f1_score(y, y_pred)  # Calculate F1 score
            print(f"logreg, acc in {name}:{acc}, f1_score:{f1}")
            f1_list.append(f1)  # Append F1 score to the list
            acc_list.append(acc)  # Append accuracy to the list
    display_average_scores()









