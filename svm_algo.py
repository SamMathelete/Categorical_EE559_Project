import torch
import numpy as np
from sklearn.svm import SVC
from utils import build_onehot_features

def svm (
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        target: int,
        K_list: list[int]
) -> float:
    d = X_train.shape[1]
    input_nodes = [i for i in range(d) if i != target]
    X_tr_oh = build_onehot_features(X_train, input_nodes, K_list).cpu().numpy()
    X_te_oh = build_onehot_features(X_test, input_nodes, K_list).cpu().numpy()
    y_train = X_train[:, target].cpu().numpy()
    y_test = X_test[:, target].cpu().numpy()
    if len(np.unique(y_train)) < 2:
        majority = int(y_train[0])
        return float((y_test == majority).mean())
    clf = SVC(kernel = "rbf", C = 1.0, gamma = "scale")
    clf.fit(X_tr_oh, y_train)
    preds = clf.predict(X_te_oh)
    return float((preds == y_test).mean())
