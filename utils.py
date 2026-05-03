import torch

def build_onehot_features (
        X: torch.Tensor,
        node_indices: list[int],
        K_list: list[int]
) -> torch.Tensor | None:
    if len(node_indices) == 0:
        return None
    N = X.shape[0]
    device = X.device
    total_dim = sum(K_list[i] for i in node_indices)
    out = torch.zeros(N, total_dim, device = device)
    offset = 0
    for i in node_indices:
        out[torch.arange(N, device = device), offset + X[:, i]] = 1.0
        offset += K_list[i]
    return out

def is_dag (
        A: torch.Tensor
) -> bool:
    d = A.shape[0]
    A_int = A.to(dtype = torch.int).cpu()
    in_degree = A_int.sum(dim = 1).tolist()
    out_neighbors = [[] for _ in range(d)]
    for i in range(d):
        for j in range(d):
            if A_int[i, j].item() == 1:
                out_neighbors[j].append(i)
    queue = [i for i in range(d) if in_degree[i] == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in out_neighbors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return visited == d

def random_dag(
        d: int, 
        device: torch.device,
        edge_prob: float = 0.4
) -> torch.Tensor:
    mask = torch.tril(torch.ones(d, d, device = device), diagonal=-1)
    A = (torch.rand(d, d, device = device) < edge_prob).int() * mask.int()
    return A

def random_weight_tensor(
        A: torch.Tensor, 
        K_list: list[int], 
        K_max: int, 
        device: torch.device,
        weight_scale: float = 1.5
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    d = A.shape[0]
    W = torch.zeros(d, d, K_max, K_max, device = device)
    b = []
    for i in range(d):
        b_i = torch.randn(K_list[i], device = device) * 0.5
        b.append(b_i)
        parents = torch.where(A[i] == 1)[0]
        for j in parents:
            W[i, j, :K_list[i], :K_list[j]] = torch.randn(
                K_list[i], K_list[j], device = device) * weight_scale
    return W, b

@torch.no_grad()
def sample_from_dag(
    A: torch.Tensor, 
    W: torch.Tensor, 
    b: torch.Tensor, 
    K_list: list[int], 
    N: int,
    device: torch.device
) -> torch.Tensor:
    d = A.shape[0]
    X = torch.zeros(N, d, dtype = torch.long, device = device)
    for i in range(d):
        parents = torch.where(A[i] == 1)[0]
        logits = b[i].unsqueeze(0).expand(N, -1).clone()
        for j in parents:
            logits += W[i, j, :K_list[i], :][:, X[:, j]].T
        probs = torch.softmax(logits, dim=1)
        X[:, i] = torch.multinomial(probs, num_samples = 1).squeeze(1)
    return X