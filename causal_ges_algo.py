import torch
import math
import torch.optim as optim
import torch.nn as nn
from utils import build_onehot_features, is_dag

def fit_cpd (
        X: torch.Tensor,
        node: int,
        parents: list[int],
        K_list: list[int]
) -> tuple[torch.Tensor, torch.Tensor, float]:
    N = X.shape[0]
    device = X.device
    K_i = K_list[node]
    y = X[:, node]
    if len(parents) == 0:
        counts = torch.zeros(K_i, device = device)
        counts.scatter_add_(0, y, torch.ones(N, device = device))
        counts.clamp_min_(1e-8)
        probs = counts / counts.sum()
        log_probs = probs.log()
        nll = -log_probs[y].sum().item()
        return None, log_probs, nll
    features = build_onehot_features(X, parents, K_list)
    feat_dim = features.shape[1]
    linear = nn.Linear(feat_dim, K_i, device = device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    optimizer = optim.LBFGS(linear.parameters(), max_iter = 30, line_search_fn = "strong_wolfe")
    loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
    
    def closure():
        optimizer.zero_grad(set_to_none = True)
        loss = loss_fn(linear(features), y)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    with torch.no_grad():
        nll = loss_fn(linear(features), y).item()
    return linear.weight.detach(), linear.bias.detach(), nll

def local_bic_score (
        X: torch.Tensor,
        node: int,
        parents: list[int],
        K_list: list[int]
) -> float:
    N = X.shape[0]
    K_i = K_list[node]
    _, _, nll = fit_cpd(X, node, parents, K_list)
    if len(parents) == 0:
        n_params = K_i - 1
    else:
        feat_dim = sum(K_list[j] for j in parents)
        n_params = K_i * feat_dim + K_i - 1
    return 2.0 * nll + n_params * math.log(N)

def ges_search (
        X: torch.Tensor,
        K_list: list[int],
        target_only: int | None = None
) -> torch.Tensor:
    d = X.shape[1]
    device = X.device
    A = torch.zeros(d, d, dtype = torch.int, device = device)
    nodes_to_search = [target_only] if target_only is not None else list(range(d))
    for i in nodes_to_search:
        current_parents = []
        current_bic = local_bic_score(X, i, current_parents, K_list)
        improved = True
        while improved:
            improved = False
            best_bic = current_bic
            best_add = None
            candidates = [j for j in range(d) if j != i and j not in current_parents]
            for j in candidates:
                A[i, j] = 1
                acyclic = is_dag(A)
                A[i, j] = 0
                if not acyclic:
                    continue
                trial_bic = local_bic_score(X, i, current_parents + [j], K_list)
                if trial_bic < best_bic:
                    best_bic = trial_bic
                    best_add = j
            if best_add is not None:
                current_parents.append(best_add)
                A[i, best_add] = 1
                current_bic = best_bic
                improved = True
        improved = True
        while improved:
            improved = False
            best_bic = current_bic
            best_remove = None
            for j in current_parents:
                trial_parents = [p for p in current_parents if p != j]
                trial_bic = local_bic_score(X, i, trial_parents, K_list)
                if trial_bic < best_bic:
                    best_bic = trial_bic
                    best_remove = j
            if best_remove is not None:
                current_parents.remove(best_remove)
                A[i, best_remove] = 0
                current_bic = best_bic
                improved = True
        for j in current_parents:
            A[i, j] = 1
    return A

def causal_inference (
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        target: int,
        K_list: list[int]
) -> tuple[float, torch.Tensor]:
    A_est = ges_search(X_train, K_list, target_only = target)
    parents = torch.where(A_est[target] == 1)[0].tolist()
    W_local, b_local, _ = fit_cpd(X_train, target, parents, K_list)
    y_test = X_test[:, target]
    if len(parents) == 0:
        mode = b_local.argmax()
        acc = (y_test == mode).float().mean().item()
    else:
        features = build_onehot_features(X_test, parents, K_list)
        logits = features @ W_local.T + b_local.unsqueeze(0)
        preds = logits.argmax(dim = 1)
        acc = (preds == y_test).float().mean().item()
    return acc, A_est
