from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import pickle
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
torch.random.manual_seed(123)
np.random.seed(123)
parser = argparse.ArgumentParser(description='Spam classifier training')
parser.add_argument('--dataset', type=str, default='trec07p', metavar='N',
                    help='trec06p, trec07p, LingSpam, Enron_Spam')



datasets = ['trec06p', 'trec07p', 'LingSpam', 'Enron_Spam']
args = parser.parse_args()

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

    logreg = LogisticRegression(random_state=123, max_iter=10000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_trec06p)  # Make predictions
    acc = accuracy_score(y_trec06p, y_pred)  # Calculate accuracy
    f1 = f1_score(y_trec06p, y_pred)  # Calculate F1 score
    print(f"logreg is trained, on Trec06p acc :{acc}, f1_score:{f1}")
    with open(f'/data/CQC/result/{filename}/logreg_model_seed{seed}.pkl', 'wb') as file:
        pickle.dump(logreg, file)