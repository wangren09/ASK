import torch
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DKNN:

    def __init__(
            self,
            model,
            train_data,
            train_targets,
            n_class=10,
            hidden_layers=-1,
            n_neighbors=5,
            metric="l2",
            batch_size=128,
            device=torch.device("cpu")
    ):

        self.hidden_layers = hidden_layers
        self._model = self._wrap_model(model)

        self.train_data = train_data
        self.train_targets = np.array(train_targets)

        self.hidden_layers = self._model.hidden_layers

        self.device = device
        self._model.eval()  # make sure the model is in the eval mode
        self._model.to(self.device)

        self.n_class = n_class
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self._nns = self._build_nns()

    def _get_hidden_repr(self, x, return_targets=False):
        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i:i + self.batch_size]
            if return_targets:
                hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
            else:
                hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            if self.metric == "cosine":
                hidden_reprs_batch = [
                    hidden_repr_batch / hidden_repr_batch.pow(2).sum(dim=1, keepdim=True).sqrt()
                    for hidden_repr_batch in hidden_reprs_batch
                ]
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [
            np.concatenate([hidden_batch[i] for hidden_batch in hidden_reprs], axis=0)
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
                    hidden_reprs.append(x.detach().cpu())
                out = self.classifier(x)
                return [hidden_reprs[i].flatten(start_dim=1) for i in self.hidden_layers], out

        return ModelWrapper(model, self.hidden_layers)

    def _build_nns(self):
        hidden_reprs, _ = self._get_hidden_repr(self.train_data)
        return [
            NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(hidden_repr)
            for hidden_repr in tqdm(hidden_reprs)
        ]

    def predict(self, x):
        hidden_reprs, _ = self._get_hidden_repr(x)
        nn_indices = [
            nn.kneighbors(hidden_repr, return_distance=False)
            for nn, hidden_repr in zip(self._nns, hidden_reprs)
        ]
        nn_indices = np.concatenate(nn_indices, axis=1)
        nn_labels = self.train_targets[nn_indices]
        nn_labels_count = np.stack(list(map(
            lambda x: np.bincount(x, minlength=10),
            nn_labels
        )))
        return nn_labels_count / len(self.hidden_layers) / self.n_neighbors

    def __call__(self, x):
        return self.predict(x)
