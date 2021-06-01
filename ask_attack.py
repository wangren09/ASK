from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import torch.nn as nn
from ask_loss import ASKLoss


class ASKAttack:
    def __init__(
            self,
            model,
            train_data,
            train_targets,
            n_class=10,
            n_neighbors=5,
            class_samp_size=None,
            eps=8 / 255,
            step_size=2 / 255,
            max_iter=10,
            random_init=True,
            metric="l2",
            batch_size=20,
            hidden_layers=-1,
            kappa=1,
            temperature=0.1,
            random_seed=1234,
            device=torch.device("cpu")
    ):
        self.class_samp_size = class_samp_size
        self.n_class = n_class
        self.metric = metric
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.device = device

        self._model = self._wrap_model(model)
        self._model.to(self.device)
        self._model.eval()

        self.hidden_layers = self._model.hidden_layers
        self.random_seed = random_seed

        self.train_data = self._samp_data(train_data, train_targets)
        self.n_neighbors = n_neighbors
        self._nns = self._build_nns()

        self.temperature = [temperature for _ in range(len(self.hidden_layers))] \
            if isinstance(temperature, (int, float)) else temperature
        self.kappa = [kappa for _ in range(len(self.hidden_layers))] \
            if isinstance(kappa, (int, float)) else kappa

        self.ask_loss = [ASKLoss(temperature=t, metric=metric, type="class-wise") for t in self.temperature]

        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_init = random_init

    def _samp_data(
            self,
            train_data,
            train_targets,
    ):
        if self.class_samp_size is None:
            return [train_data[train_targets == i] for i in range(self.n_class)]
        else:
            np.random.seed(self.random_seed)
            class_indices = []
            for i in range(self.n_class):
                inds = np.where(train_targets == i)[0]
                subset = np.random.choice(inds, size=self.class_samp_size, replace=False)
                class_indices.append(subset)
            return [train_data[subset] for subset in class_indices]

    def _get_hidden_repr(self, x, return_targets=False):
        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i:i + self.batch_size]
            with torch.no_grad():
                if return_targets:
                    hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
                else:
                    hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            if self.metric == "cosine":
                hidden_reprs_batch = [
                    hidden_repr_batch/hidden_repr_batch.pow(2).sum(dim=-1,keepdim=True).sqrt()
                    for hidden_repr_batch in hidden_reprs_batch
                ]
            hidden_reprs_batch = [hidden_repr_batch.cpu() for hidden_repr_batch in hidden_reprs_batch]
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [
            torch.cat([hidden_batch[i] for hidden_batch in hidden_reprs], dim=0)
            for i in range(len(self.hidden_layers))
        ]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.hidden_mappings = []
                start_layer = 0
                if hasattr(model, "feature"):
                    start_layer = 1
                    self.hidden_mappings.append(model.feature)
                self.hidden_mappings.extend([
                    m[1] for m in model.named_children()
                    if isinstance(m[1], nn.Sequential) and "layer" in m[0]
                ])
                if hidden_layers == -1:
                    self.hidden_layers = list(range(len(self.hidden_mappings)))
                else:
                    self.hidden_layers = hidden_layers
                self.hidden_layers = [hl + start_layer for hl in hidden_layers]
                self.classifier = self._model.classifier

            def forward(self, x):
                hidden_reprs = []
                for mp in self.hidden_mappings:
                    x = mp(x)
                    hidden_reprs.append(x)
                out = self.classifier(x)
                return [hidden_reprs[i].flatten(start_dim=1) for i in self.hidden_layers], out

        return ModelWrapper(model, self.hidden_layers)

    def _build_nns(self):
        nns = [[] for _ in range(len(self.hidden_layers))]
        for class_data in self.train_data:
            hidden_reprs, _ = self._get_hidden_repr(class_data)
            for i, hidden_repr in enumerate(hidden_reprs):
                nns[i].append(NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(hidden_repr))
        return nns

    def attack(self, x, y, x_refs, x_adv=None):
        if x_adv is None:
            if self.random_init:
                x_adv = 2 * self.eps * (torch.rand_like(x) - 0.5) + x
                x_adv = x_adv.clamp(0.0, 1.0)
            else:
                x_adv = torch.clone(x).detach()
        x_adv.requires_grad_(True)
        hidden_repr_adv, _ = self._model(x_adv)
        loss = 0
        for ask_loss, hidden_repr, x_ref, kappa in zip(self.ask_loss, hidden_repr_adv, x_refs, self.kappa):
            if self.metric == "cosine":
                hidden_repr = hidden_repr / hidden_repr.pow(2).sum(dim=1, keepdim=True).sqrt()
            loss += kappa * ask_loss(
                hidden_repr,
                y,
                x_ref.to(x),
                torch.arange(self.n_class).repeat_interleave(self.n_neighbors).to(x)
            )
        grad = torch.autograd.grad(loss, x_adv)[0]
        pert = self.step_size * grad.sign()
        x_adv = (x_adv + pert).clamp(0.0, 1.0).detach()
        pert = (x_adv - x).clamp(-self.eps, self.eps)
        return x + pert

    def _get_nns(self, x):
        hidden_reprs, _ = self._get_hidden_repr(x)
        nn_reprs = []
        for i, hidden_repr, nns in zip(range(len(self.hidden_layers)), hidden_reprs, self._nns):
            nn_inds = [torch.LongTensor(nn.kneighbors(hidden_repr, return_distance=False)) for nn in nns]
            nn_repr = [class_data[nn_ind] for class_data, nn_ind in zip(self.train_data, nn_inds)]
            nn_reprs.append(self._get_hidden_repr(torch.cat(nn_repr, dim=1).reshape(-1, *x.shape[1:]))[0][i])
        return [nn_repr.reshape(x.size(0), self.n_neighbors*self.n_class, -1) for nn_repr in nn_reprs]

    def generate(self, x, y=None):

        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(self.device)
            nn_reprs_batch = self._get_nns(x_batch)
            if y is None:
                y_batch = self._model(x_batch)
                if isinstance(y_batch, tuple):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(self.device)
            else:
                y_batch = y[i: i + self.batch_size].to(self.device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(x_batch, y_batch, nn_reprs_batch)
                else:
                    x_adv_batch = self.attack(x_batch, y_batch, nn_reprs_batch, x_adv_batch)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).cpu()


if __name__ == "__main__":
    from dknn import DKNN
    from models.vgg import VGG16
    from data_utils import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16()
    model.load_state_dict(torch.load("./checkpoints/cifar10_vgg16.pt")["model_state"])
    model.to(device)
    model.eval()

    trainloader, testloader = get_dataloaders(
        "cifar10",
        root="./datasets",
        batch_size=100,
        download=False,
        augmentation=False,
        train_shuffle=False,
        num_workers=1
    )

    train_data, train_targets = [], []
    for x, y in trainloader:
        train_data.append(x)
        train_targets.append(y)
    train_data = torch.cat(train_data, dim=0)
    train_targets = torch.cat(train_targets)

    ask_attack = ASKAttack(
        model,
        train_data,
        train_targets,
        hidden_layers=[3, ],
        class_samp_size=100,
        device=device
    )
    dknn = DKNN(
        model,
        torch.cat(ask_attack.train_data, dim=0),
        torch.arange(ask_attack.n_class).repeat_interleave(ask_attack.class_samp_size),
        hidden_layers=ask_attack.hidden_layers,
        metric=ask_attack.metric,
        device=device
    )

    x_batch, y_batch = next(iter(testloader))
    pred_dknn_clean = dknn.predict(x_batch)
    print("Clean accuracy of DkNN is {}".format(
        (pred_dknn_clean.argmax(axis=1) == y_batch.numpy()).astype("float").mean()
    ))
    x_adv = ask_attack.generate(x_batch, y_batch)
    pred_dknn_adv = dknn.predict(x_adv)
    print("Adversarial accuracy of DkNN is {}".format(
        (pred_dknn_adv.argmax(axis=1) == y_batch.numpy()).astype("float").mean()
    ))
