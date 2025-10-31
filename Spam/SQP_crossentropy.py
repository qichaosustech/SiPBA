import os
import pickle
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import torch.nn as nn
import numpy as np
import argparse
import torch.autograd.functional as F
import sys
import logging
from datetime import datetime
import os


log_dir = "log_test"
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
    X = torch.tensor(X_test, dtype=torch.float64)
    y = torch.tensor(y_test, dtype=torch.float64)
    logits = X @ w
    preds = torch.where(logits > 0, 1.0, -1.0)
    f1 = f1_score(y.numpy(), preds.numpy())
    acc = (preds == y).sum().item() / len(y)
    return f1, acc

def pack(w, tau):
    return torch.cat([w, tau])

def unpack(x):
    d = w_dim
    return x[:d], x[d:]

def objective_np(x_np, X, y, lambda_upper=0.01):
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    w, tau = unpack(x)
    z = X @ w
    fw = z + tau * torch.sum(w ** 2)
    criterion = nn.CrossEntropyLoss()
    log_loss = criterion(fw, y)
    reg = (lambda_upper / 2) * torch.sum(w ** 2)
    loss = log_loss + reg
    return loss.item()

def constraint_np(x_np, X):
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    w, tau = unpack(x)
    z = X @ w
    fw = z + tau * torch.sum(w ** 2)
    y = torch.ones_like(z)
    criterion = nn.CrossEntropyLoss()
    log_loss = criterion(fw, y)
    val = tau + 0.001 * log_loss
    return val.detach().numpy()

def constraint_jac_np(x_np, X):
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    def fun(xx):
        w, tau = unpack(xx)
        z = X @ w
        fw = z + tau * torch.sum(w ** 2)
        y = torch.ones_like(z)
        criterion = nn.CrossEntropyLoss()
        log_loss = criterion(fw, y)
        val = tau + 0.001 * log_loss
        return val
    J = F.jacobian(fun, x)
    return J.detach().numpy()

def objective_jac_np(x_np, X, y, lambda_upper=0.01):
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    def fun(xx):
        w, tau = unpack(xx)
        z = X @ w
        fw = z + tau * torch.sum(w ** 2)
        criterion = nn.CrossEntropyLoss()
        log_loss = criterion(fw, y)
        reg = (lambda_upper / 2) * torch.sum(w ** 2)
        loss = log_loss + reg
        return loss
    g = F.jacobian(fun, x)
    return g.detach().numpy().ravel()



torch.random.manual_seed(123)
np.random.seed(123)
parser = argparse.ArgumentParser(description='Spam classifier training')
parser.add_argument('--dataset', type=str, default='trec06p', metavar='N',
                    help='trec06p, trec07p, LingSpam, Enron_Spam')



datasets = ['trec06p', 'trec07p', 'LingSpam', 'Enron_Spam']
args = parser.parse_args()


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

    X = torch.tensor(X_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    y = (y+torch.ones_like(y))/2
    n, d = X.shape
    w_dim = d
    x0 = np.zeros(d + n, dtype=np.float64)
    constraint = {
        'type': 'eq',
        'fun': lambda x: constraint_np(x, X),
        'jac': lambda x: constraint_jac_np(x, X)
    }

    res = minimize(
        fun=lambda x: objective_np(x, X, y),
        x0=x0,
        jac=lambda x: objective_jac_np(x, X, y),
        constraints=[constraint],
        method='trust-constr',
        options={'maxiter': 500, 'verbose': 2}
    )

    x_opt = torch.tensor(res.x, dtype=torch.float64)
    w_opt, tau_opt = unpack(x_opt)
    w_opt = torch.tensor(w_opt, dtype=torch.float64)
    f1, acc = eval_model(X_test, y_test, w_opt)
    print(f"F1={f1:.4f}, Acc={acc:.4f}")
    torch.save(w_opt, f'/data/CQC/result/{filename}/model_SQP_CE_weights_seed{seed}.pth')

