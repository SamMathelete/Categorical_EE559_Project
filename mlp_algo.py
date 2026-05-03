import torch
import torch.nn as nn
import torch.optim as optim
from utils import build_onehot_features

class SimpleMLP (nn.Module):
    def __init__ (
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)    
        )
    def forward(self, x):
        return self.net(x)

def train_mlp (
        model: SimpleMLP,
        X_oh: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 500,
        lr = 1e-2
) -> SimpleMLP:
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none = True)
        loss = loss_fn(model(X_oh), y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predict_mlp(
        model: SimpleMLP,
        X_oh: torch.Tensor
) -> torch.Tensor:
    return model(X_oh).argmax(dim = 1)

def supervised_MLP (
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        target: int,
        K_list: list[int],
        hidden_dim: int = 10
) -> float:
    d = X_train.shape[1]
    device = X_train.device
    X_test = X_test.to(device = device)
    input_nodes = [i for i in range(d) if i != target]
    input_dim = sum(K_list[i] for i in input_nodes)
    output_dim = K_list[target]
    X_tr_oh = build_onehot_features(X_train, input_nodes, K_list)
    X_te_oh = build_onehot_features(X_test, input_nodes, K_list)
    y_train = X_train[:, target]
    y_test = X_test[:, target]
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device = device)
    model = train_mlp(model, X_tr_oh, y_train, epochs = 500, lr = 1e-2)
    preds = predict_mlp(model, X_te_oh)
    return (preds == y_test).float().mean().item()
