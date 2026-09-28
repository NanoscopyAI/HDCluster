# %% [markdown]
# # Clustering speed + ARI / IoU evaluation — Scenarios 4 & 9 & Hela & LifeActin
# 




# %%
import os
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError          # scipy >= 1.8
except ImportError:
    from scipy.spatial.qhull import QhullError    # older scipy
from matplotlib.path import Path as MplPath
from sklearn.metrics import adjusted_rand_score
import h5py
import tables

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hdcluster import hdcluster

from sklearn.cluster import DBSCAN

from dbscan import DBSCAN as DBSCAN_p


# Root of the ARI-and-IoU repo (the folder that contains the "Ground Truth - Scenario N" dirs).
REPO = Path("/path/to/ARI-and-IoU-cluster-analysis-evaluation/repo")

# Which scenarios to evaluate.
SCENARIOS = [4, 9]

# IoU rasterisation. The R source uses a (ROIsize + 400) pixel mask at ~1 nm resolution
# (ROIsize = 2000 -> 2400). That is accurate but slow; drop MASK_DIM for faster runs and
# raise it toward 2400 when you want R-equivalent fidelity.
ROI_SIZE = 2000
MASK_DIM = 600            # grid resolution (pixels per side) used for IoU masks
MIN_CLUSTER_PTS = 3       # result clusters need >= 3 points to form a polygon (matches R)

# Optional: only process the first N ground-truth files per scenario (None = all 50).
N_FILES = None
# Detailed benchmark results are overwritten on each completed run so they can
# be loaded later with ``pd.read_csv(RESULTS_CSV)``.
RESULTS_CSV =  Path("hdc_speedtest_results.csv")

# 3. Load scenarios 4 and 9

# %%
def scenario_dir(n):
    return REPO / f"Ground Truth - Scenario {n}"

def _fileno(path):
    """Trailing integer in a filename, e.g. lowden_MoleculeList_10.csv -> 10."""
    return int(Path(path).stem.split("_")[-1])

def list_gt_files(n):
    files = glob.glob(str(scenario_dir(n) / "*.csv"))
    return sorted(files, key=_fileno)

def load_molecule_list(path):
    """Read one ground-truth CSV -> DataFrame with columns x, y, index."""
    return pd.read_csv(path)

gt_files = {}
for n in SCENARIOS:
    files = list_gt_files(n)
    if not files:
        raise FileNotFoundError(f"No CSVs found in {scenario_dir(n)}")
    gt_files[n] = files
    ex = load_molecule_list(files[0])
    print(f"Scenario {n}: {len(files)} files | example {os.path.basename(files[0])} "
          f"({len(ex)} molecules, cols={list(ex.columns)})")



# Adding real data to the scenarios
# %%
hela_dir = "/path/to/hela/data"

actin_dir = "/path/to/actin/data"


# %%
# Test Params
db_eps_4 = 50 # for ARI 
db_min_pts_4 = 4 # for ARI

db_eps_9 = 45 # for ARI 
db_min_pts_9 = 12 # for ARI

mth_4=59
mth_9=77

# real data
mth_actin = 6 #[1,6,12,18]
mth_hela = 120 #[10, 20, 80, 120]

eps_actin = mth_actin
mnpts_actin = 3 #[3, 5, 10]

eps_hela = mth_hela
mnpts_hela = 3 #[3, 5, 10]

# %%

def cluster_dbscan(coords, scenario):
    if scenario == 4: 
        print(f"DBSCAN: eps={db_eps_4}, min_samples={db_min_pts_4} on scenario {scenario}")
        labels = DBSCAN(eps=db_eps_4, min_samples=db_min_pts_4).fit_predict(coords)
        
    elif scenario == "actin":
        print(f"DBSCAN: eps={eps_actin}, min_samples={mnpts_actin} on scenario {scenario}")
        labels = DBSCAN(eps=eps_actin, min_samples=mnpts_actin).fit_predict(coords)
        
    elif scenario == "hela":
        print(f"DBSCAN: eps={eps_hela}, min_samples={mnpts_hela} on scenario {scenario}")
        labels = DBSCAN(eps=eps_hela, min_samples=mnpts_hela).fit_predict(coords)
        
    else: 
        print(f"DBSCAN: eps={db_eps_9}, min_samples={db_min_pts_9} on scenario {scenario}")
        labels = DBSCAN(eps=db_eps_9, min_samples=db_min_pts_9).fit_predict(coords)
        
    return labels+1


def cluster_p_dbscan(coords, scenario):
    """

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)  -- x, y of each molecule.

    Returns
    -------
    labels : np.ndarray, shape (N,)    -- integer label per molecule
                                          (0 = noise, 1..K = clusters).
    """
    
    if scenario == 4: 
        print(f"Parallel DBSCAN: eps={db_eps_4}, min_samples={db_min_pts_4} on scenario {scenario}")
        labels, _ = DBSCAN_p(coords, eps=db_eps_4, min_samples=db_min_pts_4)
    elif scenario == "actin":
        print(f"Parallel DBSCAN: eps={eps_actin}, min_samples={mnpts_actin} on scenario {scenario}")
        labels, _ = DBSCAN_p(coords, eps=eps_actin, min_samples=mnpts_actin)
    elif scenario == "hela":
        print(f"Parallel DBSCAN: eps={eps_hela}, min_samples={mnpts_hela} on scenario {scenario}")
        labels, _ = DBSCAN_p(coords, eps=eps_hela, min_samples=mnpts_hela)
    else: 
        print(f"Parallel DBSCAN: eps={db_eps_9}, min_samples={db_min_pts_9} on scenario {scenario}")
        labels, _ = DBSCAN_p(coords, eps=db_eps_9, min_samples=db_min_pts_9)
    return labels+1

def cluster_hdc(coords, scenario):
    """Run HDC clustering on the given coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)  -- x, y of each molecule.

    Returns
    -------
    labels : np.ndarray, shape (N,)    -- integer label per molecule
                                          (0 = noise, 1..K = clusters).
    """
    if scenario == 4: 
        print(f"HDC Python: mth={mth_4}, beta=1.9 on scenario {scenario}")
        _, labels = hdcluster(coords, mth_4, True, beta=1.9, workers=-1)
    elif scenario == "actin":
        print(f"HDC Python: mth={mth_actin}, beta=0.2 on scenario {scenario}")
        _, labels = hdcluster(coords, mth_actin, True, beta=0.2, workers=-1)
    elif scenario == "hela":
        print(f"HDC Python: mth={mth_hela}, beta=0.2 on scenario {scenario}")
        _, labels = hdcluster(coords, mth_hela, True, beta=0.2, workers=-1)
    else: 
        print(f"HDC Python: mth={mth_9}, beta=0.6 on scenario {scenario}")  
        _, labels = hdcluster(coords, mth_9, True, beta=0.6, workers=-1)

    return labels

_MATLAB_ENG = None

def matlab_engine():
    """Start the MATLAB engine once and reuse it.

    Starting it costs a few seconds and leaks a MATLAB process if never quit,
    so it must happen once, outside the timed region --
    otherwise every hdc_m timing is just MATLAB's startup cost. The engine also
    starts in this notebook's folder, so the repo root (where hdcluster.m lives)
    has to be added to MATLAB's search path explicitly.
    """
    global _MATLAB_ENG
    if _MATLAB_ENG is None:
        import matlab.engine
        _MATLAB_ENG = matlab.engine.start_matlab()
        _MATLAB_ENG.addpath(str(PROJECT_ROOT), nargout=0)
        # Warm up: the first call pays MATLAB's JIT cost, which would otherwise
        # land on whichever file happens to be timed first.
        warm = matlab.double(np.random.default_rng(0).normal(0, 10, (200, 2)).tolist())
        _MATLAB_ENG.hdcluster(warm, mth_4, 1, nargout=2)
    return _MATLAB_ENG

def cluster_hdc_matlab(coords, scenario):
    import matlab
    eng = matlab_engine()

    is_noise = 1
    if scenario == 4: 
        print(f"HDC MATLAB: mth={mth_4}, beta=1.9 on scenario {scenario}")
        _, labels = eng.hdcluster(matlab.double(coords.tolist()), mth_4, 1.9 ,is_noise, nargout=2)
    elif scenario == "actin":
        print(f"HDC MATLAB: mth={mth_actin}, beta=0.2 on scenario {scenario}")
        _, labels = eng.hdcluster(matlab.double(coords.tolist()), mth_actin, 0.2 ,is_noise, nargout=2)
    elif scenario == "hela":
        print(f"HDC MATLAB: mth={mth_hela}, beta=0.2 on scenario {scenario}")
        _, labels = eng.hdcluster(matlab.double(coords.tolist()), mth_hela, 0.2 ,is_noise, nargout=2)
    else:   
        print(f"HDC MATLAB: mth={mth_9}, beta=0.6 on scenario {scenario}")
        _, labels = eng.hdcluster(matlab.double(coords.tolist()), mth_9, 0.6 ,is_noise, nargout=2)
    
    return labels

# Register every algorithm you want to benchmark here.
CLUSTERERS = {
    "dbscan": cluster_dbscan,
    "p_dbscan": cluster_p_dbscan,
    "hdc_py": cluster_hdc,
    "hdc_m":  cluster_hdc_matlab,
}

def run_timed(fn, coords, scenario):
    """Run a clustering function and return (labels, elapsed_seconds)."""
    t0 = time.perf_counter()
    labels = fn(coords, scenario)
    elapsed = time.perf_counter() - t0
    labels = np.asarray(labels).astype(int).ravel()
    assert labels.shape[0] == coords.shape[0], \
        f"clustering must return one label per point ({labels.shape[0]} != {coords.shape[0]})"
    return labels, elapsed

#  5. Adjusted Rand Index (ARI)


# %%
def ari_score(gt_index, result_labels):
    """Adjusted Rand Index between ground-truth and clustering labels."""
    return adjusted_rand_score(np.asarray(gt_index), np.asarray(result_labels))


#  6. Intersection over Union (IoU)

# %%
def _cluster_union_mask(coords, labels, grid_pts, grid_shape, min_pts):
    """Boolean union of the convex-hull polygons of every qualifying cluster,
    rasterised onto grid_pts (an (M, 2) array of pixel-centre coordinates)."""
    mask = np.zeros(grid_pts.shape[0], dtype=bool)
    for lab in np.unique(labels):
        if lab <= 0:
            continue
        member = coords[labels == lab]
        if member.shape[0] < max(min_pts, 3):
            continue  # need >= 3 points for a polygon (matches R's zero-mask case)
        try:
            hull = ConvexHull(member)
        except (QhullError, ValueError):
            continue  # collinear / degenerate cluster -> no area
        poly = MplPath(member[hull.vertices])
        mask |= poly.contains_points(grid_pts)
    return mask.reshape(grid_shape)


def iou_score(coords, gt_index, result_labels, mask_dim=MASK_DIM, min_pts=MIN_CLUSTER_PTS):
    """Area IoU of ground-truth cluster regions vs result cluster regions."""
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, mask_dim),
                         np.linspace(ymin, ymax, mask_dim))
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    shape = gx.shape

    gt_mask  = _cluster_union_mask(coords, np.asarray(gt_index),     grid_pts, shape, min_pts=1)
    res_mask = _cluster_union_mask(coords, np.asarray(result_labels), grid_pts, shape, min_pts=min_pts)

    intersection = np.logical_and(gt_mask, res_mask).sum()
    union        = np.logical_or(gt_mask, res_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


#  7. Run the evaluation
# 
# Loops over scenarios 4 & 9 and every registered algorithm, recording **time**, **ARI**, and **IoU**
# per file. This cell raises `NotImplementedError` until you fill in the clustering placeholder above.

# %%
records = []
for n in SCENARIOS:
    files = gt_files[n]
    if N_FILES:
        files = files[:N_FILES]
    for path in files:
        df = load_molecule_list(path)
        coords = df[["x", "y"]].to_numpy(float)
        gt_index = df["index"].to_numpy(int)
        for name, fn in CLUSTERERS.items():
            labels, elapsed = run_timed(fn, coords, n)
            records.append({
                "scenario":  n,
                "file":      os.path.basename(path),
                "algorithm": name,
                "n_points":  coords.shape[0],
                "time_s":    elapsed,
                "ARI":       ari_score(gt_index, labels),
                "IoU":       iou_score(coords, gt_index, labels),
                "status":    "completed",
            })

#  Iterate over the real datasets

hela_df = pd.read_csv(hela_dir, header=None, names=["x", "y", "z"])

actin_df = pd.read_hdf(actin_dir, key='locs')
actin_df = actin_df[["x", "y", "z"]]
scale_nm_per_pix = 130
actin_df[["x", "y"]] *= scale_nm_per_pix

coords = hela_df[["x", "y", "z"]].to_numpy(float)
for name, fn in CLUSTERERS.items():
    _, elapsed = run_timed(fn, coords, "hela")
    records.append({
        "scenario":  "hela",
        "algorithm": name,
        "n_points":  coords.shape[0],
        "time_s":    elapsed,
        "ARI":       np.nan,
        "IoU":       np.nan,
        "status":    "completed",
    })

coords = actin_df[["x", "y", "z"]].to_numpy(float)
for name, fn in CLUSTERERS.items():
    if name in ["p_dbscan", "hdc_m"]:
        print("Skipping Parallel DBSCAN on actin (65.9M points)")
        records.append({
            "scenario":  "actin",
            "algorithm": name,
            "n_points":  coords.shape[0],
            "time_s":    np.nan,
            "ARI":       np.nan,
            "IoU":       np.nan,
            "status":    "skipped",
        })
        continue

    _, elapsed = run_timed(fn, coords, "actin")
    records.append({
        "scenario":  "actin",
        "algorithm": name,
        "n_points":  coords.shape[0],
        "time_s":    elapsed,
        "ARI":       np.nan,
        "IoU":       np.nan,
        "status":    "completed",
    })

# 
results = pd.DataFrame.from_records(records)
results.to_csv(RESULTS_CSV, index=False)
print(f"Saved {len(results)} benchmark results to {RESULTS_CSV}")


#  8. Summary  —  mean speed / ARI / IoU per scenario & algorithm

# %%
if len(results):
    summary = (results
               .groupby(["scenario", "algorithm"])[["time_s", "n_points"]]
               .agg(["mean"]))
    display(summary)
else:
    print("Error in algorithms.")



