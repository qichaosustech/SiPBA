import os
from networks import LeNet5Feats, classifier
import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import argparse
import higher
import hypergrad as hg
import time
from torchvision.datasets import FashionMNIST
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.random.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='Bilevel Training')
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST'])
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='hyperlogs')
args = parser.parse_args()
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# Load dataset
def load_data(dataset_name):
    if dataset_name == 'MNIST':
        full_train_data = MNIST(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    else:
        full_train_data = FashionMNIST(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]))

    train_data, val_data = random_split(full_train_data, [50000, 10000])

    test_data = MNIST(args.data, train=False, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])) if dataset_name == 'MNIST' else FashionMNIST(args.data, train=False, download=True,
                                                     transform=transforms.Compose([
                                                         transforms.Resize((32, 32)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.2860,), (0.3530,))
                                                     ]))

    data_train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
    data_val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=4)
    data_test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    return data_train_loader, data_val_loader, data_test_loader


# Define the experiment
def run_experiment(experiment_id, data_train_loader, data_val_loader, data_test_loader):

    hypernet = LeNet5Feats().cuda()
    cnet = classifier(n_features=84, n_classes=10).cuda()
    data_train_iter = iter(data_train_loader)
    data_val_iter = iter(data_val_loader)

    fhnet = higher.monkeypatch(hypernet, copy_initial_weights=True).cuda()
    hparams = list(hypernet.parameters())
    hparams = [hparam.requires_grad_(True) for hparam in hparams]
    fcnet = higher.monkeypatch(cnet, copy_initial_weights=True).cuda()
    params = list(cnet.parameters())
    params = [param.requires_grad_(True) for param in params]

    criterion = torch.nn.CrossEntropyLoss().cuda()


    def outer_loss(params, hparams, more=False):

        nonlocal data_val_iter

        try:
            images, labels = next(data_val_iter)
        except StopIteration:
            data_val_iter = iter(data_val_loader)
            images, labels = next(data_val_iter)

        images, labels = images.cuda(), labels.cuda()

        feats = fhnet(images, params=hparams)
        outputs = fcnet(feats, params=params)
        loss = criterion(outputs, labels)

        preds = outputs.data.max(1)[1]
        correct = preds.eq(labels.data.view_as(preds)).sum()
        acc = float(correct) / labels.size(0)

        if more:
            return loss, acc
        else:
            return loss


    def inner_loss(params, hparams, more=False):
        nonlocal data_train_iter
        try:
            images, labels = next(data_train_iter)
        except StopIteration:
            data_train_iter = iter(data_train_loader)
            images, labels = next(data_train_iter)

        images, labels = images.cuda(), labels.cuda()

        feats = fhnet(images, params=hparams)
        outputs = fcnet(feats, params=params)
        loss = criterion(outputs, labels)

        preds = outputs.data.max(1)[1]
        correct = preds.eq(labels.data.view_as(preds)).sum()
        acc = float(correct) / labels.size(0)

        if more:
            return loss, acc
        else:
            return loss

    def map_func(params, hparams):
        g = inner_loss(params, hparams)
        grads = torch.autograd.grad(g, params, create_graph=True)
        return [w - lr * g for w, g in zip(params, grads)]

    def inner_solver(params, hparams, steps, params0=None):
        params_new = [p.requires_grad_(True) for p in params]
        optim = torch.optim.Adam(params=params_new, lr=lr)
        for i in range(steps):
            optim.zero_grad()
            g = inner_loss(params_new, hparams)
            g.backward()
            optim.step()
        return [p.detach().clone() for p in params_new]

    def evaluate(params, hparams, data_test_loader):
        fhnet.eval()
        fcnet.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in data_test_loader:
                images, labels = images.cuda(), labels.cuda()
                feats = fhnet(images, params=hparams)
                outputs = fcnet(feats, params=params)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)  
                preds = outputs.argmax(dim=1)
                total_correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    if args.dataset == 'MNIST':
        T = 20
    else:
        T = 30
    lr=0.001
    outer_opt = torch.optim.Adam(hparams, lr=lr)
    steps = 10000
    total_time = 0
    running_time, test_accs, test_losses = [], [], []
    loss, acc = evaluate(params, hparams, data_test_loader)
    running_time.append(total_time)
    test_accs.append(acc)
    test_losses.append(loss)
    print('loss={}, acc={},time={}'.format(loss, acc, total_time))

    for step in range(1, steps + 1):

        t0 = time.time()
        params = inner_solver(params, hparams, steps=T)
        outer_opt.zero_grad()
        _, cost = hg.fixed_point(params, hparams, T, map_func, outer_loss, set_grad=True)
        outer_opt.step()
        t1 = time.time() - t0
        total_time += t1
        # count += 1
        if step % 20 == 0:
            loss, acc = evaluate(params, hparams, data_test_loader)
            running_time.append(total_time)
            test_accs.append(acc)
            test_losses.append(loss)
            print('step={}, loss={}, acc={},time={}'.format(step, loss, acc, total_time))
        if step == 200:
            lr = 0.0001
            for param_group in outer_opt.param_groups:
                param_group['lr'] = lr
        if total_time > 300:
            loss, acc = evaluate(params, hparams, data_test_loader)
            running_time.append(total_time)
            test_accs.append(acc)
            test_losses.append(loss)
            print('step={}, loss={}, acc={},time={}'.format(step, loss, acc, total_time))
            break

    return running_time, test_accs, test_losses


# Run 5 experiments and collect statistics
final_results = {
    "mean_loss": [],
    "std_loss": [],
    "mean_acc": [],
    "std_acc": [],
    "mean_time": [],
    "std_time": []
}

for exp_id in range(1, 11):
    print(f"Running experiment {exp_id}...")

    data_train_loader, data_val_loader, data_test_loader = load_data(args.dataset)
    running_time, test_accs, test_losses = run_experiment(exp_id, data_train_loader, data_val_loader, data_test_loader)

    final_results["mean_loss"].append(test_losses)
    final_results["std_loss"].append(test_losses)
    final_results["mean_acc"].append(test_accs)
    final_results["std_acc"].append(test_accs)
    final_results["mean_time"].append(running_time)
    final_results["std_time"].append(running_time)

np.save(f"result/{args.dataset}_averaged_results_AID-FP.npy", final_results)
print("Experiment finished!")
