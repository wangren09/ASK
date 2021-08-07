import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
from pgd import PGD
from dknn import DKNN
import numpy as np
import os
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from ask_loss import ASKLoss
from models.vgg import VGG16
from models.resnet import ResNet18
from data_utils import get_dataloaders, get_transforms, GenericDataset

parser = ArgumentParser(description="PyTorch ASK training")
parser.add_argument("--epochs", help="the number of training epochs", type=int, default=200)
parser.add_argument("--hidden-layer", help="specify the layer to use for ask training", type=int, default=3)
parser.add_argument("--include-self", help="whether to include self in positive pairs", action="store_true")
parser.add_argument("--no-adv", help="whether not to use adversarial reference data", action="store_true")
parser.add_argument("--no-mixing", help="whether not to mix clean data with their adversarial copies",
                    action="store_true")
parser.add_argument("--n-ref", help="the number of reference data per class", type=int, default=5)
parser.add_argument("--beta", help="balancing factor between ce and ask loss", type=float, default=1.0)
parser.add_argument("--root", help="the path to dataset directory", default="./datasets")
parser.add_argument("--num-eval", help="the number of batches to evaluate", type=int, default=4)
parser.add_argument("--no-cuda", help="whether to use cuda", action="store_true")
parser.add_argument("--dknn-size", help="the number of training data used for dknn evaluation", type=int, default=20000)
parser.add_argument("--batch-size", help="batch size for training", type=int, default=128)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
parser.add_argument("--momentum", "-m", type=float, default=0.9)
parser.add_argument("--weight-decay", "-w", type=float, default=0.0002)
parser.add_argument("--eps-ask", help="maximum perturbation of ASK attack", type=int, default=8)
parser.add_argument("--eps-train", help="maximum perturbation of PGD attack during training", type=int,
                    default=4)
parser.add_argument("--eps-eval", help="maximum perturbation of PGD attack during evaluation", type=int,
                    default=8)
parser.add_argument("--step-size", help="step size of each attack", type=int, default=2)
parser.add_argument("--max-iter", help="maximum iterations for ask attacks", type=int, default=10)
parser.add_argument("--metric", help="distance metric for ask loss and dknn", default="cosine")
parser.add_argument("--no-random-init", help="whether not to use random initialization in ask attack",
                    action="store_true")
parser.add_argument("--temperature", help="scaling factor for ask loss", type=float, default=0.1)
parser.add_argument("--seed", help="random seed for reproducibility", type=int, default=3)
parser.add_argument("--n-class", help="number of classes in the classification problem", type=int, default=10)
parser.add_argument("--dataset", help="which dataset to use", choices=["cifar10", "imagenette"])
parser.add_argument("--checkpoint", "-c", help="temporary checkpoint", default="./checkpoints/ask_train.pt")
parser.add_argument("--output", "-o", help="output log file", default="./logs/ask_train.log")
parser.add_argument("--download", "-d", help="whether to download the dataset", action="store_true")
parser.add_argument("--disable-ask", help="disable ask training and use the standard adversarial training",
                    action="store_true")
parser.add_argument("--num-workers", help="set number of subprocesses in data loading", type=int, default=4)

args = parser.parse_args()

root = args.root
download = not os.path.exists(os.path.join(root, "cifar-10-batches-py"))
device = torch.device("cpu" if args.no_cuda else "cuda")
train_size_for_eval = args.dknn_size
batch_size = args.batch_size
epochs = args.epochs
temperature = args.temperature
metric = args.metric
n_class = args.n_class
dataset = args.dataset
download = args.download
hidden_layer = args.hidden_layer
num_eval = args.num_eval
disable_ask = args.disable_ask

# load data
trainloader, testloader = get_dataloaders(
    dataset, root=root, download=download, batch_size=batch_size, augmentation=True, num_workers=args.num_workers)

if dataset == "cifar10":
    model = VGG16()
if dataset == "imagenette":
    model = ResNet18()

# dknn training data
trainloader_dknn, _ = get_dataloaders(
    dataset, root=root, download=download, batch_size=batch_size, augmentation=False, num_workers=args.num_workers)
train_data = []
train_targets = []
for x, y in trainloader_dknn:
    train_data.append(x)
    train_targets.append(y)
train_data = torch.cat(train_data, dim=0)
train_targets = torch.cat(train_targets, dim=0)

np.random.seed(args.seed)  # fix training data
dknn_indices = []
for i in range(n_class):
    class_inds = np.where(train_targets == i)[0]
    choices = np.random.choice(len(class_inds), size=args.dknn_size//n_class, replace=False)
    dknn_indices.append(class_inds[choices])
dknn_indices = np.concatenate(dknn_indices)
train_data_dknn = train_data[dknn_indices]
train_targets_dknn = train_targets[dknn_indices]

# instantiate the losses
loss_fn_ce = nn.CrossEntropyLoss()
loss_fn_cknn = ASKLoss(temperature=temperature, metric=metric)

# pgd for training and evaluation
pgd_train = PGD(
    eps=args.eps_train/255,
    step_size=args.step_size/255,
    batch_size=batch_size,
    loss_fn=loss_fn_ce
)
pgd_eval = PGD(
    eps=args.eps_eval/255,
    step_size=args.step_size/255,
    batch_size=batch_size,
    loss_fn=loss_fn_ce
)

# hyperparameters for training
include_self = args.include_self
ref_advaug = not args.no_adv
mixed = not args.no_mixing
max_iter = args.max_iter
eps = args.eps_ask/255
step_size = args.step_size/255
c = args.beta
n_ref = args.n_ref

# reference dataloader for each class
if not disable_ask:
    transform_train, _ = get_transforms(dataset, True)
    refloader = [DataLoader(GenericDataset(
        train_data[train_targets == i], i, transform_train
    ), shuffle=True, batch_size=n_ref) for i in range(10)]

# keep track of best validation results and model weights
best_acc = 0
state_dict = None

# initialize the model or restart from a previous checkpoint
checkpoint_path = args.checkpoint
if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))
model.to(device)
optimizer = SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True
)
print("Searching for checkpoint...")
if os.path.exists(checkpoint_path):
    print("Checkpoint found!")
    print("Loading the checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    print("Last epoch: {}".format(checkpoint["epoch"]))
    print(f"Best adversarial accuracy of DkNN on hidden layer {hidden_layer} is {checkpoint['best_acc']}")
    print("Restarting training from the loaded checkpoint")
    best_acc = checkpoint["best_acc"]
    last_epoch = checkpoint["epoch"] - 1
    train_epochs = epochs - checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
else:
    print("No checkpoint found! Random initializing...")
    last_epoch = -1
    train_epochs = epochs

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=last_epoch)

for e in range(train_epochs):
    train_loss_clean = 0
    train_correct_clean = 0
    train_loss_ask = 0
    train_loss_adv = 0
    train_correct_adv = 0
    train_total = 0
    model.train()
    ref_data = iter(zip(*refloader))
    with tqdm(trainloader, desc=f"{last_epoch + 2 + e}/{epochs} epochs") as t:
        for i, (x, y) in enumerate(t):
            if not disable_ask:
                x_ref = torch.cat(next(ref_data), dim=0)
                # renew the reference dataloader
                if x_ref.size(0) != n_class*n_ref:
                    ref_data = iter(zip(*refloader))
                    x_ref = torch.cat(next(ref_data), dim=0)
                y_ref = torch.LongTensor(range(10)).repeat_interleave(n_ref)
                if ref_advaug:
                    if mixed:
                        x_ref = torch.cat([
                            x_ref,
                            pgd_train.generate(model, x_ref, y_ref, device=device)
                        ], dim=0)
                        y_ref = y_ref.repeat_interleave(2)
                    else:
                        x_ref = pgd_train.generate(model, x_ref, y_ref, device=device)
                model.eval()
                with torch.no_grad():
                    out_hd, out = model(x.to(device), extra_out=hidden_layer)
                    loss_ce = loss_fn_ce(out, y.to(device))
                model.train()
                out_hdref, _ = model(x_ref.to(device), extra_out=hidden_layer)
                train_loss_clean += loss_ce.item() * x.size(0)
                train_correct_clean += (out.max(dim=1)[1] == y.to(device)).sum().item()
                train_total += x.size(0)
                if args.no_random_init:
                    x_hdadv = x.clone().detach()
                else:
                    x_hdadv = x.clone().detach() + (2 * torch.rand_like(x) - 1) * eps
                for _ in range(max_iter):
                    x_hdadv.requires_grad_(True)
                    out_hdadv, _ = model(x_hdadv.to(device), extra_out=hidden_layer)
                    if include_self:
                        loss_cknn = loss_fn_cknn(out_hdadv, y.to(device), out_hdref, y_ref.to(device), out_hd)
                    else:
                        loss_cknn = loss_fn_cknn(out_hdadv, y.to(device), out_hdref, y_ref.to(device))
                    grad = torch.autograd.grad(loss_cknn, x_hdadv)[0].detach()
                    x_hdadv = (x_hdadv.data + step_size * grad.sign()).clamp(0, 1)
                    x_hdadv = ((x_hdadv - x).clamp(-eps, eps) + x).detach()
                x_adv = pgd_train.generate(model, x, y, device=device)
                model.train()
                _, out_adv = model(x_adv.to(device), extra_out=hidden_layer)
                out_hd, _ = model(x.to(device), extra_out=hidden_layer)
                loss_ce = loss_fn_ce(out_adv.to(device), y.to(device))
                out_hdadv, _ = model(x_hdadv.to(device), extra_out=hidden_layer)
                if include_self:
                    loss_cknn = loss_fn_cknn(out_hdadv, y.to(device), out_hdref, y_ref.to(device), out_hd)
                else:
                    loss_cknn = loss_fn_cknn(out_hdadv, y.to(device), out_hdref, y_ref.to(device))
                loss = loss_ce + c * loss_cknn
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_ask += loss.item() * x.size(0)  # ASK Advsersarial Soft K-nearest neighbor loss
            else:
                model.eval()
                with torch.no_grad():
                    out = model(x.to(device))
                    loss_ce = loss_fn_ce(out, y.to(device))
                    train_loss_clean += loss_ce.item() * x.size(0)
                    train_correct_clean += (out.max(dim=1)[1] == y.to(device)).sum().item()
                x_adv = pgd_train.generate(model, x, y, device=device)
                model.train()
                out_adv = model(x_adv.to(device))
                loss = loss_fn_ce(out_adv, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_adv += loss.item() * x.size(0)
                train_correct_adv += (out_adv.max(dim=1)[1] == y.to(device)).sum().item()
                train_total += x.size(0)
            if i < len(trainloader) - 1:
                if not disable_ask:
                    t.set_postfix({
                        "train_loss_clean": train_loss_clean / train_total,
                        "train_acc_clean": train_correct_clean / train_total,
                        "train_loss_ask": train_loss_ask / train_total,
                    })
                else:
                    t.set_postfix({
                        "train_loss_clean": train_loss_clean / train_total,
                        "train_acc_clean": train_correct_clean / train_total,
                        "train_loss_adv": train_loss_adv / train_total,
                        "train_acc_adv": train_correct_adv / train_total,
                    })
            else:
                scheduler.step()
                dknn = DKNN(
                    model,
                    train_data_dknn,
                    train_targets_dknn,
                    hidden_layers=[hidden_layer, ],
                    device=device,
                    metric=metric,

                )
                test_correct_dknn = 0
                test_loss_clean = 0
                test_correct_clean = 0
                test_loss_rob = 0
                test_correct_rob = 0
                test_total = 0
                model.eval()
                for _, (x, y) in zip(range(num_eval), testloader):
                    with torch.no_grad():
                        out = model(x.to(device))
                        loss = loss_fn_ce(out, y.to(device))
                    test_loss_clean += loss.item() * x.size(0)
                    test_correct_clean += (out.max(dim=1)[1] == y.to(device)).sum().item()
                    test_total += x.size(0)
                    x_adv = pgd_eval.generate(model, x, y, device=device)
                    with torch.no_grad():
                        out = model(x_adv.to(device))
                        loss = loss_fn_ce(out, y.to(device))
                    test_loss_rob += loss.item() * x.size(0)
                    test_correct_rob += (out.max(dim=1)[1] == y.to(device)).sum().item()
                    pred = dknn(x_adv).argmax(axis=1)
                    test_correct_dknn += (pred == y.numpy()).sum()
                if not disable_ask:
                    t.set_postfix({
                        "train_loss_clean": train_loss_clean / train_total,
                        "train_acc_clean": train_correct_clean / train_total,
                        "train_loss_ask": train_loss_ask / train_total,
                        "test_loss_clean": test_loss_clean / test_total,
                        "test_acc_clean": test_correct_clean / test_total,
                        "test_loss_pgdrob": test_loss_rob / test_total,
                        "test_acc_pgdrob": test_correct_rob / test_total,
                        "test_acc_dknn": test_correct_dknn / test_total
                    })
                else:
                    t.set_postfix({
                        "train_loss_clean": train_loss_clean / train_total,
                        "train_acc_clean": train_correct_clean / train_total,
                        "train_loss_adv": train_loss_adv / train_total,
                        "train_acc_adv": train_correct_adv / train_total,
                        "test_loss_clean": test_loss_clean / test_total,
                        "test_acc_clean": test_correct_clean / test_total,
                        "test_loss_pgdrob": test_loss_rob / test_total,
                        "test_acc_pgdrob": test_correct_rob / test_total,
                        "test_acc_dknn": test_correct_dknn / test_total
                    })
                if test_correct_dknn / test_total > best_acc and test_correct_clean / test_total > 0.83:
                    best_acc = test_correct_dknn / test_total
                    state_dict = model.state_dict()
                    epoch = e + 1
                    optimizer_state = optimizer.state_dict()
                    torch.save({
                        "model_state": state_dict,
                        "epoch": epoch,
                        "best_acc": best_acc,
                        "optimizer_state": optimizer_state,
                    }, checkpoint_path)
