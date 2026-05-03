import os
import numpy as np
import pandas as pd
import torch
import time
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from causal_ges_algo import causal_inference
from mlp_algo import supervised_MLP
from svm_algo import svm

matplotlib.use('Agg')

DEVICE = torch.device("cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ASIA_NODES = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
ASIA_D = 8
ASIA_K_LIST = [2] * 8
ASIA_CSV_PATH = 'asia_data.csv'


def load_asia_dataset(csv_path: str = ASIA_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded ASIA dataset from cache: {csv_path}  (N = {len(df)})")
    assert list(df.columns) == ASIA_NODES, (
        f"Column order mismatch: expected {ASIA_NODES}, got {list(df.columns)}"
    )
    for c in df.columns:
        assert set(df[c].unique()).issubset({0, 1}), (
            f"Column {c} has non-binary values: {df[c].unique()}"
        )
    return df


def df_to_tensor(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    return torch.tensor(df.values, dtype = torch.long, device = device)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman',
                                'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


COLORS = {
    'Method 1: Supervised MLP':  '#636363',
    'Method 2: SVM (RBF)':       '#cb6e2c',
    'Method 3: Causal Inference':'#238b45',
}
MARKERS = {
    'Method 1: Supervised MLP':  'o',
    'Method 2: SVM (RBF)':       '^',
    'Method 3: Causal Inference':'s',
}
DASHES = {
    'Method 1: Supervised MLP':  (4, 2),
    'Method 2: SVM (RBF)':       (6, 1),
    'Method 3: Causal Inference':(1, 0),
}
LABELS = {
    'Method 1: Supervised MLP':  'Supervised MLP',
    'Method 2: SVM (RBF)':       'SVM (RBF)',
    'Method 3: Causal Inference':'Causal Inference (GES + CPD)',
}

def plot_accuracy_vs_n(train_sizes, summary, save_path = 'asia_acc_vs_n.pdf'):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    N = np.array(train_sizes)

    for name in summary:
        means = np.array(summary[name]['means'])
        stds = np.array(summary[name]['stds'])
        ax.errorbar(
            N, means, yerr = stds,
            color = COLORS[name], marker = MARKERS[name],
            dashes = DASHES[name], label = LABELS[name],
            capsize = 3, capthick = 1.0, elinewidth = 1.0,
            markeredgecolor = 'white', markeredgewidth = 0.6,
            zorder = 3,
        )

    ax.set_xscale('log')
    ax.set_xlabel('Training samples ($N$)')
    ax.set_ylabel('Classification accuracy')
    ax.set_xticks(train_sizes)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.tick_params(which = 'both', top = True, right = True)
    ax.minorticks_off()

    all_m, all_s = [], []
    for name in summary:
        all_m.extend(summary[name]['means'])
        all_s.extend(summary[name]['stds'])
    y_lo = min(m - s for m, s in zip(all_m, all_s))
    y_hi = max(m + s for m, s in zip(all_m, all_s))
    pad = 0.05 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.legend(frameon = True, fancybox = False, edgecolor = '#cccccc',
              framealpha = 0.9, loc = 'lower right')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def save_results(train_sizes, summary, path = 'asia_results.json'):
    with open(path, 'w') as f:
        json.dump({
            'train_sizes': train_sizes,
            'methods': summary,
        }, f, indent = 2)
    print(f"Results saved: {path}")


if __name__ == "__main__":
    K_list = ASIA_K_LIST
    target = 6  
    df = load_asia_dataset()
    X_all = df_to_tensor(df, device = DEVICE)
    N_total = X_all.shape[0]
    N_test = 2000
    assert N_total > N_test, f"Need N_total > N_test; got {N_total} <= {N_test}"
    perm = torch.randperm(N_total)
    test_idx = perm[:N_test]
    pool_idx = perm[N_test:]
    X_test = X_all[test_idx]
    X_pool = X_all[pool_idx]
    N_pool = X_pool.shape[0]

    print(f"Dataset: ASIA (fixed, N_total = {N_total})")
    print(f"  Test set:        N_test = {N_test}")
    print(f"  Training pool:   N_pool = {N_pool}")
    print(f"  Target:          {ASIA_NODES[target]} (idx {target})")
    p_target_yes = (X_test[:, target] == 0).float().mean().item()
    print(f"  Test marginal:   P({ASIA_NODES[target]} = yes) = {p_target_yes:.3f}\n")

    train_sizes = [30, 60, 120, 250, 500, 1000, 2000, 4000]
    n_repeats = 10
    train_sizes = [n for n in train_sizes if n <= N_pool]

    results = {name: {n: [] for n in train_sizes} for name in [
        'Method 1: Supervised MLP',
        'Method 2: SVM (RBF)',
        'Method 3: Causal Inference',
    ]}

    t_start = time.time()

    for N in train_sizes:
        for rep in range(n_repeats):
            sub = torch.randperm(N_pool)[:N]
            X_train = X_pool[sub]

            results['Method 1: Supervised MLP'][N].append(
                supervised_MLP(X_train, X_test, target, K_list))
            results['Method 2: SVM (RBF)'][N].append(
                svm(X_train, X_test, target, K_list))
            acc3, _ = causal_inference(X_train, X_test, target, K_list)
            results['Method 3: Causal Inference'][N].append(acc3)

        m1 = np.mean(results['Method 1: Supervised MLP'][N])
        m2 = np.mean(results['Method 2: SVM (RBF)'][N])
        m3 = np.mean(results['Method 3: Causal Inference'][N])
        print(f"N={N:5d} | MLP={m1:.3f}  SVM={m2:.3f}  Causal={m3:.3f}")

    elapsed = time.time() - t_start
    print(f"\nSweep time: {elapsed:.1f}s")

    summary = {}
    for name in results:
        means = [float(np.mean(results[name][n])) for n in train_sizes]
        stds = [float(np.std(results[name][n])) for n in train_sizes]
        summary[name] = {'means': means, 'stds': stds}

    print("\n" + "=" * 60)
    print("Accuracy summary:")
    for name, vals in summary.items():
        print(f"\n{name}:")
        for i, n in enumerate(train_sizes):
            print(f"  N={n:5d}: {vals['means'][i]:.3f} +/- {vals['stds'][i]:.3f}")

    save_results(train_sizes, summary, path = 'asia_results.json')
    plot_accuracy_vs_n(train_sizes, summary, save_path = 'asia_acc_vs_n.pdf')
    print("\nDone.")