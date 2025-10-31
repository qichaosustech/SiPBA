from sklearn.decomposition import PCA
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


def weighted_norm(X1, X0, A):
    diff = torch.matmul(X1 - X0,A)
    return torch.sum((diff) ** 2)


def upper_loss_logistic(w, X1, y, lambda_upper=0.01):
    y=(y+torch.ones_like(y))/2
    z = torch.matmul(X1, w)
    criterion = nn.CrossEntropyLoss()
    log_loss = criterion(z, y)
    regularization = (lambda_upper / 2) * torch.sum(w ** 2)
    loss = log_loss + regularization
    return loss
def lower_loss_logistic(w, X1, X0, lambda_lower=0.1):
    z = torch.matmul(X1, w)
    y = torch.ones_like(z)
    criterion = nn.CrossEntropyLoss()
    log_loss = criterion(z, y)
    weighted_norm_loss = (lambda_lower / 2) * weighted_norm(X1, X0, A)
    loss = log_loss + weighted_norm_loss
    return loss
def upper_loss_svm(w, X1, y, lambda_upper=0.01):
    z = torch.matmul(X1, w)
    hinge_loss = torch.mean(torch.clamp(1 - y * z, min=0))
    regularization = (lambda_upper / 2) * torch.sum(w ** 2)
    loss = hinge_loss + regularization
    return loss
def lower_loss_svm(w, X1, X0, lambda_lower=0.1):
    z = torch.matmul(X1, w)
    hinge_loss = torch.mean(torch.clamp(1 - z, min=0))
    loss = hinge_loss + (lambda_lower / 2) * weighted_norm(X1, X0, A)
    return loss
def aggeration_loss(w, X1, X2, X0, y, rho, lambda_upper=1e-6, lambda_lower=0.1):
    loss = upper_loss_svm(w, X1, y, lambda_upper=lambda_upper) - rho * (
            lower_loss_svm(w, X1, X0, lambda_lower=lambda_lower) - lower_loss_svm(w, X2, X0, lambda_lower=lambda_lower))
    return loss
def aggeration_logistic_loss(w, X1, X2, X0, y, rho, lambda_upper=1e-6, lambda_lower=0.1):
    loss = upper_loss_logistic(w, X1, y, lambda_upper=lambda_upper) - rho * (
            lower_loss_logistic(w, X1, X0, lambda_lower=lambda_lower) - lower_loss_logistic(w, X2, X0, lambda_lower=lambda_lower))
    return loss
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
parser.add_argument('--dataset', type=str, default='trec07p', metavar='N',
                    help='trec06p, trec07p, LingSpam, Enron_Spam')

datasets = ['trec06p', 'trec07p', 'LingSpam', 'Enron_Spam']
args = parser.parse_args()
for seed in range(10):
    train_data=args.dataset
    filename=f"trained_by_{train_data}"
    X_train, y_train, X_test, y_test,vectorizer=generate_data(train_data, train_size=500, filename= filename,seed=seed)
    a=np.sum(y_train==1)+np.sum(y_test==1)
    b=np.sum(y_train==-1)+np.sum(y_test==-1)
    print(f"ham{a},spam{b}")
    globals()[f'X_{train_data}'] = X_test
    globals()[f'y_{train_data}'] = y_test
    for test_data in [dataset for dataset in datasets if dataset != train_data]:
        X, y = generate_data(test_data, vectorizer=vectorizer, filename=filename,seed=seed)
        globals()[f'X_{test_data}'] = X
        globals()[f'y_{test_data}'] = y
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')





    if args.dataset == 'trec06p':
        alpha_0 = 0.03
        beta_0 = 1e-6
        epochs = 10000
    elif args.dataset == 'trec07p':
        alpha_0 = 0.1
        beta_0 = 1e-2
        epochs = 20000
    elif args.dataset == 'LingSpam':
        alpha_0 = 0.02
        beta_0 = 5e-5
        epochs = 10000
    elif args.dataset == 'Enron_Spam':
        alpha_0 = 0.02
        beta_0 = 1e-7
        epochs = 20000


    w = torch.zeros(X_train.shape[1], requires_grad=True, device=device)
    X0 = torch.tensor(X_train, dtype=torch.float32, requires_grad=False).to(device)  # Convert to tensor and move to device
    X1 = torch.tensor(X0.detach().cpu().numpy(), dtype=torch.float32, requires_grad=True,
                      device=device)  # X1 as a tensor with gradient enabled
    X2 = torch.tensor(X0.detach().cpu().numpy(), dtype=torch.float32, requires_grad=True,
                      device=device)  # X2 as a tensor with gradient enabled
    y = torch.tensor(y_train, dtype=torch.float32).to(device)  # y tensor on the device

    X0_pca = X0.detach().cpu().numpy()
    pca = PCA(n_components=100)
    X0_pca = pca.fit_transform(X0_pca)
    A = torch.tensor(pca.components_.T, dtype=torch.float32, device=device)

    for i in range(epochs):

        p = 0.01
        q = 0.01
        alpha =alpha_0* (i+ 1) ** (-8 * p - 8* q)
        beta = beta_0 * (i+ 1) ** (-2*p-q)
        eta = beta_0 * (i+ 1) ** (-2*p-q)
        rho = 10 * (i+ 1) ** p
        delta = 1e-6 * (i+ 1) ** (-q)
        loss_X1 = aggeration_loss(w, X1, X2, X0, y, rho)-2*delta*torch.sum(X1*X2)
        grad_X1 = torch.autograd.grad(loss_X1, X1)[0]
        # Normalize gradient
        if torch.norm(grad_X1)>0:
            grad_X1=grad_X1/torch.norm(grad_X1)
        loss_X2 = aggeration_loss(w, X1, X2, X0, y, rho) + delta * torch.sum((X2 - X1) ** 2)
        grad_X2 = torch.autograd.grad(loss_X2, X2)[0]
        # Normalize gradient
        if torch.norm(grad_X2) > 0:
            grad_X2 = grad_X2 / torch.norm(grad_X2)
        X1.data += beta * grad_X1

        X2.data -= eta * grad_X2
        loss = aggeration_loss(w, X1, X2, X0, y, rho)
        grad_w = torch.autograd.grad(loss, w)[0]
        # Normalize gradient
        if torch.norm(grad_w) > 0:
            grad_w = grad_w / torch.norm(grad_w)
        w.data -=alpha * grad_w
    f1, accuracy=eval_model( X_trec06p, y_trec06p,w)
    print(f"On Trec06: f1={f1}, accuracy={accuracy}")
    torch.save(w, f'/data/CQC/result/{filename}/model_weights_seed{seed}.pth')


    if args.dataset == 'trec06p':
        alpha_0 = 0.1
        beta_0 = 1e-4
        epochs = 10000
    elif args.dataset == 'trec07p':
        alpha_0 = 0.05
        beta_0 = 1e-4
        epochs = 10000
    elif args.dataset == 'LingSpam':
        alpha_0 = 0.05
        beta_0 = 1e-7
        epochs = 10000
    elif args.dataset == 'Enron_Spam':
        alpha_0 = 0.01
        beta_0 = 1e-7
        epochs = 10000


    w_2 = torch.zeros(X_train.shape[1], requires_grad=True, device=device)
    # Convert the data to tensors and move them to the appropriate device
    X1 = torch.tensor(X0.detach().cpu().numpy(), dtype=torch.float32, requires_grad=True,
                      device=device)  # X1 as a tensor with gradient enabled
    X2 = torch.tensor(X0.detach().cpu().numpy(), dtype=torch.float32, requires_grad=True,
                      device=device)  # X2 as a tensor with gradient enabled
    y = torch.tensor(y_train, dtype=torch.float32).to(device)  # y tensor on the device

    for i in range(epochs):
        p = 0.01
        q = 0.01
        alpha = alpha_0 * (i + 1) ** (-8* p - 8* q)
        beta = beta_0 * (i + 1) ** (-2*p-q)
        eta = beta_0 * (i + 1) ** (-2*p-q)
        rho = 10 * (i + 1) ** p
        delta = 1e-6 * (i + 1) ** (-q)
        loss_X1 = aggeration_logistic_loss(w_2, X1, X2, X0, y, rho)-2*delta*torch.sum(X1*X2)
        grad_X1 = torch.autograd.grad(loss_X1, X1)[0]
        if torch.norm(grad_X1)>0:
            grad_X1 = grad_X1 / torch.norm(grad_X1)
        loss_X2 = aggeration_logistic_loss(w_2, X1, X2, X0, y, rho) + delta * torch.sum((X2 - X1) ** 2)
        grad_X2 = torch.autograd.grad(loss_X2, X2)[0]
        if torch.norm(grad_X2)>0:
            grad_X2 = grad_X2 / torch.norm(grad_X2)

        X1.data += beta * grad_X1


        X2.data -= eta * grad_X2
        loss = aggeration_logistic_loss(w_2, X1, X2, X0, y, rho)
        grad_w = torch.autograd.grad(loss, w_2)[0]
        # Normalize gradient
        if torch.norm(grad_w) > 0:
            grad_w = grad_w / torch.norm(grad_w)
        w_2.data -= alpha * grad_w
    f1, accuracy = eval_model( X_trec06p, y_trec06p,w_2)
    print(f"On Trec06: f1={f1}, accuracy={accuracy}")
    torch.save(w_2, f'/data/CQC/result/{filename}/model_2_weights_seed{seed}.pth')
