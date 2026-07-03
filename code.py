"""
CGS410 Project: Topological Constraints of Natural Language DAGs
Vivek Bansiwal (241182)

Full pipeline: Data Acquisition -> Graph Construction -> Baseline Generation
-> Statistical Comparison (two-sample Kolmogorov-Smirnov tests)

Requires: conllu, numpy, scipy, matplotlib, requests
    pip install conllu numpy scipy matplotlib requests --break-system-packages
"""

import os
import random
from collections import deque, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from conllu import parse_incr

try:
    import requests
except ImportError:
    requests = None

# =========================================================================
# STAGE 1: DATA ACQUISITION
# =========================================================================
# Universal Dependencies treebank files (CoNLL-U format), one per language.
# Raw GitHub URLs for the *-ud-train.conllu files (auto-downloaded below).
UD_TREEBANK_URLS = {
    "English": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu",
    "Japanese": "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-train.conllu",
    "Turkish": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu",
    "Arabic": "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu",
}

DATA_DIR = "data"
UD_TREEBANK_PATHS = {
    lang: os.path.join(DATA_DIR, os.path.basename(url))
    for lang, url in UD_TREEBANK_URLS.items()
}

MIN_SENT_LEN = 3
MAX_SENT_LEN = 50
SAMPLE_SIZE = 5000  # sentences per language


def download_treebanks(paths=UD_TREEBANK_PATHS, urls=UD_TREEBANK_URLS):
    """
    Download each UD treebank file if it isn't already present locally.
    Requires the `requests` package (pip install requests).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if requests is None:
        raise RuntimeError("Please `pip install requests` to auto-download treebanks, "
                            "or manually place the .conllu files under ./data/")

    for lang, path in paths.items():
        if os.path.exists(path):
            print(f"[{lang}] already downloaded -> {path}")
            continue
        url = urls[lang]
        print(f"[{lang}] downloading {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"[{lang}] saved -> {path} ({len(resp.content)/1024:.0f} KB)")


def load_sentences(conllu_path: str, min_len=MIN_SENT_LEN, max_len=MAX_SENT_LEN,
                    sample_size=SAMPLE_SIZE, seed=42) -> list[list[dict]]:
    """
    Stage 1: Parse a CoNLL-U treebank file and return a filtered, reproducible
    random sample of sentences.

    Each returned sentence is a list of token dicts, each with (at minimum):
        {"id": int, "form": str, "head": int}
    where head == 0 denotes the root.
    """
    sentences = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            # Keep only well-formed tokens (skip multiword/ellipsis tokens
            # whose id is a tuple/range, e.g. "3-4")
            toks = [
                {"id": tok["id"], "form": tok["form"], "head": tok["head"]}
                for tok in tokenlist
                if isinstance(tok["id"], int) and tok["head"] is not None
            ]
            if min_len <= len(toks) <= max_len:
                sentences.append(toks)

    rng = random.Random(seed)
    if len(sentences) > sample_size:
        sentences = rng.sample(sentences, sample_size)
    return sentences


# =========================================================================
# STAGE 2: GRAPH CONSTRUCTION & METRICS
# =========================================================================

def build_dag(sentence: list[dict]):
    """
    Build adjacency structure for a dependency DAG.

    Returns
    -------
    nodes     : list of node ids
    children  : dict[node_id] -> list of child node ids
    root_id   : id of the root node (head == 0's dependent)
    """
    children = defaultdict(list)
    nodes = [tok["id"] for tok in sentence]
    root_id = None

    for tok in sentence:
        if tok["head"] == 0:
            root_id = tok["id"]
        else:
            children[tok["head"]].append(tok["id"])

    # Ensure every node has an (possibly empty) entry
    for n in nodes:
        _ = children[n]

    return nodes, children, root_id


# ── Metric 1: Arity ──────────────────────────────────────────────────────
def compute_mean_arity(sentence: list[dict]) -> float:
    """
    Stage 2.2: Compute mean arity of the dependency DAG.

    Cognitive interpretation:
    A node of arity k requires the parser to simultaneously maintain k
    open dependency relations in working memory. Under Cowan (2001),
    working memory capacity is 4 +/- 1 chunks, so we predict a_bar ~ 1-3.
    """
    _, children, _ = build_dag(sentence)
    out_degrees = [len(children[tok["id"]]) for tok in sentence]
    non_leaf = [d for d in out_degrees if d > 0]
    if not non_leaf:
        return 0.0
    return float(np.mean(non_leaf))  # a_bar(G)


# ── Metric 2: Tree Depth ─────────────────────────────────────────────────
def compute_tree_depth(sentence: list[dict]) -> int:
    """
    Stage 2.3: Compute tree depth of the dependency DAG via BFS from the root.

    depth(v) = dist(root, v)
    d(G)     = max_{v in V} depth(v)

    Cognitive interpretation: measures degree of recursive centre-embedding.
    Miller & Chomsky (1963) showed comprehension fails beyond 2-3 levels.
    Predict d(G)_NL << d(G)_random, scaling as O(log n).
    """
    _, children, root_id = build_dag(sentence)
    if root_id is None:
        return 0

    max_depth = 0
    queue = deque([(root_id, 0)])
    visited = set()

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        max_depth = max(max_depth, depth)
        for child in children[node]:
            if child not in visited:
                queue.append((child, depth + 1))

    return max_depth  # d(G)


# ── Metric 3: Graph Density ──────────────────────────────────────────────
def compute_graph_density(sentence: list[dict]) -> float:
    """
    Stage 2.4: Compute graph density of the dependency DAG.

    rho(G) = |E| / (n * (n-1))

    For a standard dependency tree: |E| = n-1  =>  rho = 1/n
    Non-projective dependencies (extra edges) push rho slightly above 1/n.
    """
    n = len(sentence)
    if n <= 1:
        return 0.0
    n_edges = sum(1 for tok in sentence if tok["head"] != 0)
    return n_edges / (n * (n - 1))  # rho(G)


# =========================================================================
# STAGE 3: RANDOM BASELINE GENERATION (size-matched via Prufer sequences)
# =========================================================================

def prufer_to_tree(prufer_seq: list[int], n: int) -> list[tuple[int, int]]:
    """
    Convert a Prufer sequence of length n-2 into an undirected tree on
    n labeled nodes (0..n-1). Returns a list of (u, v) undirected edges.
    """
    degree = [1] * n
    for node in prufer_seq:
        degree[node] += 1

    edges = []
    ptr = 0
    leaf_ptr = -1

    # Use a simple O(n log n) approach with a min-heap-like scan
    import heapq
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    for node in prufer_seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, node))
        degree[leaf] -= 1
        degree[node] -= 1
        if degree[node] == 1:
            heapq.heappush(leaves, node)

    # Final two remaining nodes with degree 1 form the last edge
    remaining = [i for i in range(n) if degree[i] == 1]
    edges.append((remaining[0], remaining[1]))
    return edges


def random_dag_same_size(n: int, seed=None) -> list[dict]:
    """
    Stage 3: Generate a random size-matched tree (as a DAG) with n nodes
    using a random Prufer sequence, then orient edges away from a random
    root to produce head/dependent structure comparable to the NL format.

    Returns a "sentence"-like list of token dicts: {"id", "form", "head"}
    compatible with compute_mean_arity / compute_tree_depth / compute_graph_density.
    """
    rng = random.Random(seed)

    if n <= 1:
        return [{"id": 1, "form": "w1", "head": 0}]

    if n == 2:
        edges = [(0, 1)]
    else:
        prufer_seq = [rng.randrange(n) for _ in range(n - 2)]
        edges = prufer_to_tree(prufer_seq, n)

    # Build undirected adjacency, then orient via BFS from a random root
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    root = rng.randrange(n)
    head_of = {root: 0}  # 0-indexed root maps to head=0 (root marker)
    visited = {root}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                head_of[v] = u + 1  # +1 to convert to 1-indexed "head" id
                queue.append(v)

    sentence = [
        {"id": i + 1, "form": f"w{i+1}", "head": head_of.get(i, 0)}
        for i in range(n)
    ]
    return sentence


def generate_random_baseline(nl_sentences: list[list[dict]], seed=42) -> list[list[dict]]:
    """
    Stage 3: For every NL sentence, generate one size-matched random DAG.
    """
    rng = random.Random(seed)
    return [
        random_dag_same_size(len(sent), seed=rng.randrange(10**9))
        for sent in nl_sentences
    ]


# =========================================================================
# STAGE 4: STATISTICAL ANALYSIS
# =========================================================================

def ks_test(nl_vals: np.ndarray, rand_vals: np.ndarray,
            alpha: float = 0.05) -> dict:
    """
    Two-sample Kolmogorov-Smirnov test.

    H0 : F_NL(x) = F_rand(x)              [same underlying distribution]
    H1 : F_NL(x) > F_rand(x)              [NL distribution stochastically smaller]
         i.e. NL metric values are systematically lower

    D = sup_x | F_hat_NL(x) - F_hat_rand(x) |
    H0 is rejected when p-value < alpha.
    """
    ks_stat, p_value = sp_stats.ks_2samp(
        nl_vals, rand_vals, alternative="less"  # one-sided: NL < random
    )
    return {
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "rejected": p_value < alpha,
        "nl_mean": float(np.mean(nl_vals)),
        "nl_std": float(np.std(nl_vals, ddof=1)),
        "rand_mean": float(np.mean(rand_vals)),
        "rand_std": float(np.std(rand_vals, ddof=1)),
        "delta": float(np.mean(nl_vals) - np.mean(rand_vals)),
    }


# =========================================================================
# PLOTTING (Figures 1-3 in the report)
# =========================================================================

def plot_metric_comparison(nl_by_lang: dict, rand_by_lang: dict, metric_name: str,
                            xlabel: str, ref_line=None, save_path=None):
    """
    Recreates Figures 1-3: side-by-side density histograms of NL vs Random
    baseline for each language.
    """
    langs = list(nl_by_lang.keys())
    fig, axes = plt.subplots(1, len(langs), figsize=(4.5 * len(langs), 4), sharey=False)
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        nl_vals = nl_by_lang[lang]
        rand_vals = rand_by_lang[lang]

        ax.hist(nl_vals, bins=30, density=True, alpha=0.6, label="NL", color="#4C72B0")
        ax.hist(rand_vals, bins=30, density=True, alpha=0.6, label="Random", color="#DD8452")

        if ref_line is not None:
            ax.axvline(ref_line, color="gray", linestyle="--", label=f"ref={ref_line}")
        else:
            ax.axvline(np.mean(nl_vals), color="#4C72B0", linestyle="--",
                        label=f"NL mean={np.mean(nl_vals):.3f}")
            ax.axvline(np.mean(rand_vals), color="#DD8452", linestyle="--",
                        label=f"Rand mean={np.mean(rand_vals):.3f}")

        ax.set_title(lang)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.suptitle(f"{metric_name}: Natural Language vs Random DAGs", fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def run_pipeline(alpha: float = 0.05):
    download_treebanks()  # fetch .conllu files into ./data/ if not already present

    results = {"Mean Arity": {}, "Tree Depth": {}, "Graph Density": {}}
    nl_arity, rand_arity = {}, {}
    nl_depth, rand_depth = {}, {}
    nl_density, rand_density = {}, {}

    for lang, path in UD_TREEBANK_PATHS.items():
        print(f"[{lang}] loading treebank...")
        nl_sentences = load_sentences(path)
        rand_sentences = generate_random_baseline(nl_sentences)

        # Compute metrics
        nl_a = np.array([compute_mean_arity(s) for s in nl_sentences])
        rand_a = np.array([compute_mean_arity(s) for s in rand_sentences])

        nl_d = np.array([compute_tree_depth(s) for s in nl_sentences])
        rand_d = np.array([compute_tree_depth(s) for s in rand_sentences])

        nl_rho = np.array([compute_graph_density(s) for s in nl_sentences])
        rand_rho = np.array([compute_graph_density(s) for s in rand_sentences])

        nl_arity[lang], rand_arity[lang] = nl_a, rand_a
        nl_depth[lang], rand_depth[lang] = nl_d, rand_d
        nl_density[lang], rand_density[lang] = nl_rho, rand_rho

        # Run KS tests (Table A6)
        results["Mean Arity"][lang] = ks_test(nl_a, rand_a, alpha)
        results["Tree Depth"][lang] = ks_test(nl_d, rand_d, alpha)
        results["Graph Density"][lang] = ks_test(nl_rho, rand_rho, alpha)

    # Print Table A6-style summary
    print("\n=== Table A6 — Two-Sample KS Test Results ===")
    for metric, per_lang in results.items():
        for lang, r in per_lang.items():
            print(f"{metric:15s} | {lang:10s} | D={r['ks_stat']:.4f} | "
                  f"p={r['p_value']:.4g} | reject H0: {r['rejected']}")

    # Recreate Figures 1-3
    plot_metric_comparison(nl_arity, rand_arity, "Mean Arity", "Mean Arity a_bar",
                            ref_line=3, save_path="figure1_arity.png")
    plot_metric_comparison(nl_depth, rand_depth, "Tree Depth", "Tree Depth d",
                            save_path="figure2_depth.png")
    plot_metric_comparison(nl_density, rand_density, "Graph Density", "Graph Density rho",
                            save_path="figure3_density.png")

    return results


if __name__ == "__main__":
    run_pipeline()
