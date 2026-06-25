"""
experiment_utils.py — Reusable facility-location helper functions.

Supports publication_experiments.ipynb for the CLAIO 2026 paper:
"Spectral-clustering p-median model for OXXO store placement."

All heavy-duty computation lives here so the notebook stays narrative-first.
"""
import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pulp

warnings.filterwarnings("ignore")


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_required_columns(df: pd.DataFrame, required: list, df_name: str = "DataFrame") -> None:
    """Raise ValueError listing any missing required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_experiment_data(config: dict) -> dict:
    """Load all pre-processed data for an experiment from config paths.

    Parameters
    ----------
    config : dict
        Entry from EXPERIMENT_CONFIGS. Must contain 'processed_dir'.

    Returns
    -------
    dict with keys:
        data_condensed, avg_nn_m, df_I, df_P, df_E, df_J,
        df_A, df_conflictos, j_to_p_map, n_E, node_info
    """
    processed_dir = Path(config["processed_dir"])

    # ── condensado.csv ────────────────────────────────────────────────────────
    condensado_path = processed_dir / "condensado.csv"
    if not condensado_path.exists():
        raise FileNotFoundError(
            f"condensado.csv not found at {condensado_path}.\n"
            "Run the preprocessing notebook (f_l_notebook_revised.ipynb) first."
        )
    data_condensed = pd.read_csv(condensado_path)
    data_condensed["node_id"]    = data_condensed["node_id"].astype(int)
    data_condensed["graph_node"] = (
        pd.to_numeric(data_condensed["graph_node"], errors="coerce").astype("Int64")
    )
    validate_required_columns(
        data_condensed,
        ["lat", "lon", "segmento", "POBTOT", "oxxo_presente",
         "MCA1", "MCA1_weight", "es_candidato", "node_id", "graph_node"],
        "condensado.csv",
    )

    # ── avg_nn_m.json ─────────────────────────────────────────────────────────
    nn_path = processed_dir / "avg_nn_m.json"
    if not nn_path.exists():
        raise FileNotFoundError(
            f"avg_nn_m.json not found at {nn_path}.\n"
            "Run the preprocessing notebook first."
        )
    with open(nn_path) as f:
        avg_nn_m = json.load(f)["avg_nn_m"]

    # ── node lookup dicts ─────────────────────────────────────────────────────
    _node_info = data_condensed.set_index("node_id")[
        ["lat", "lon", "graph_node", "MCA1", "POBTOT", "MCA1_weight"]
    ].copy()
    node_to_lat    = _node_info["lat"].to_dict()
    node_to_lon    = _node_info["lon"].to_dict()
    node_to_gnode  = _node_info["graph_node"].to_dict()
    node_to_mca1   = _node_info["MCA1"].to_dict()
    node_to_pobtot = _node_info["POBTOT"].to_dict()

    # ── pmedian bundle ────────────────────────────────────────────────────────
    bundle_path = processed_dir / "pmedian_bundle.parquet"
    if not bundle_path.exists():
        bundle_path = processed_dir / "pmedian_bundle.csv"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"pmedian_bundle not found in {processed_dir}.\n"
            "Run the preprocessing notebook first."
        )
    bundle = (
        pd.read_parquet(bundle_path)
        if bundle_path.suffix == ".parquet"
        else pd.read_csv(bundle_path)
    )

    _I_rows = bundle[bundle["kind"] == "I"].dropna(subset=["i"]).copy()
    _P_rows = bundle[bundle["kind"] == "P"].dropna(subset=["p"]).copy()
    _A_rows = bundle[bundle["kind"] == "A"].dropna(subset=["i", "p"]).copy()
    _C_rows = bundle[bundle["kind"] == "C"].dropna(subset=["j", "k"]).copy()

    # df_I: demand nodes ────────────────────────────────────────────────────────
    df_I = _I_rows[["i", "w", "uid_i", "cluster_i"]].copy()
    df_I.rename(columns={"uid_i": "node_id"}, inplace=True)
    df_I[["i", "cluster_i"]] = df_I[["i", "cluster_i"]].astype(int)
    df_I["node_id"]    = df_I["node_id"].astype(int)
    df_I["w"]          = df_I["w"].astype(float)
    df_I["lat"]        = df_I["node_id"].map(node_to_lat)
    df_I["lon"]        = df_I["node_id"].map(node_to_lon)
    df_I["graph_node"] = df_I["node_id"].map(node_to_gnode)
    df_I["MCA1"]       = df_I["node_id"].map(node_to_mca1)
    df_I["POBTOT"]     = df_I["node_id"].map(node_to_pobtot)
    df_I = df_I.reset_index(drop=True)

    # df_P: all facilities (existing + candidates) ─────────────────────────────
    df_P = _P_rows[["p", "uid_p", "es_existente", "cluster_p"]].copy()
    df_P.rename(columns={"uid_p": "node_id"}, inplace=True)
    df_P["p"]            = df_P["p"].astype(int)
    df_P["node_id"]      = df_P["node_id"].astype(int)
    df_P["cluster_p"]    = df_P["cluster_p"].fillna(-1).astype(int)
    df_P["es_existente"] = df_P["es_existente"].astype(bool)
    df_P["lat"]          = df_P["node_id"].map(node_to_lat)
    df_P["lon"]          = df_P["node_id"].map(node_to_lon)
    df_P["graph_node"]   = df_P["node_id"].map(node_to_gnode)
    df_P = df_P.reset_index(drop=True)

    df_E = df_P[df_P["es_existente"]].copy().reset_index(drop=True)
    df_J = df_P[~df_P["es_existente"]].copy().reset_index(drop=True)
    n_E  = len(df_E)
    df_J["j"]  = (df_J["p"].astype(int) - n_E)
    j_to_p_map = dict(zip(df_J["j"].astype(int), df_J["p"].astype(int)))

    # df_A: arc table (demand × facility, network dist ≤ D_MAX) ───────────────
    df_A = _A_rows[["i", "p", "dist_m"]].copy()
    df_A[["i", "p"]] = df_A[["i", "p"]].astype(int)
    df_A["dist_m"]   = df_A["dist_m"].astype(float)
    df_A = df_A.reset_index(drop=True)

    # df_conflictos: candidate pairs within S_MIN ──────────────────────────────
    if len(_C_rows) > 0:
        df_conflictos = _C_rows[["j", "k", "dist_m"]].copy()
        df_conflictos[["j", "k"]] = df_conflictos[["j", "k"]].astype(int)
        df_conflictos["dist_m"]   = df_conflictos["dist_m"].astype(float)
    else:
        df_conflictos = pd.DataFrame(columns=["j", "k", "dist_m"])

    return {
        "data_condensed": data_condensed,
        "avg_nn_m":        avg_nn_m,
        "df_I":            df_I,
        "df_P":            df_P,
        "df_E":            df_E,
        "df_J":            df_J,
        "df_A":            df_A,
        "df_conflictos":   df_conflictos,
        "j_to_p_map":      j_to_p_map,
        "n_E":             n_E,
        "node_info":       _node_info,
    }


# ─── Allocation ───────────────────────────────────────────────────────────────

def largest_remainder_allocation(cluster_demands: dict, total_budget: int) -> dict:
    """Largest-remainder allocation: proportional with integer rounding.

    Every cluster gets at least 1. Sum equals total_budget exactly.

    Parameters
    ----------
    cluster_demands : dict  {cluster_id: demand_weight}
    total_budget    : int   total openings to allocate

    Returns
    -------
    dict {cluster_id: n_openings}
    """
    total_w = sum(cluster_demands.values())
    if total_w <= 0:
        raise ValueError("Total cluster demand is zero — cannot allocate.")

    raw    = {c: float(cluster_demands[c]) / total_w * total_budget for c in cluster_demands}
    floors = {c: max(1, int(v)) for c, v in raw.items()}
    diff   = total_budget - sum(floors.values())

    remainders = sorted(raw, key=lambda c: raw[c] - int(raw[c]), reverse=(diff > 0))
    for c in remainders[: abs(diff)]:
        floors[c] = max(1, floors[c] + (1 if diff > 0 else -1))

    return {int(c): int(v) for c, v in floors.items()}


def compute_pnew_per_cluster(
    df_I: pd.DataFrame,
    cluster_ids: list,
    mode: str,
    pnew_fixed: int,
    pnew_total: int,
    demand_col: str = "w",
) -> dict:
    """Allocate new openings across clusters.

    Parameters
    ----------
    mode : 'fixed' | 'proportional' (by block count) | 'demand_weighted' (by ∑demand_col)

    Returns
    -------
    dict {cluster_id: n_openings}
    """
    if mode == "fixed":
        return {int(c): int(pnew_fixed) for c in cluster_ids}

    if mode == "proportional":
        demands = (
            df_I.groupby("cluster_i").size()
            .reindex(cluster_ids, fill_value=0).astype(float).to_dict()
        )
    elif mode == "demand_weighted":
        demands = (
            df_I.groupby("cluster_i")[demand_col].sum()
            .reindex(cluster_ids, fill_value=0.0).to_dict()
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'fixed', 'proportional', or 'demand_weighted'.")

    return largest_remainder_allocation(demands, pnew_total)


# ─── MILP solver — single cluster ────────────────────────────────────────────

def solve_cluster_model(
    cluster_id: int,
    df_I: pd.DataFrame,
    df_J: pd.DataFrame,
    df_P: pd.DataFrame,
    df_A: pd.DataFrame,
    df_conflictos: pd.DataFrame,
    j_to_p_map: dict,
    p_new_per_cluster: int = 7,
    penalty_uncovered: float = 5_000.0,
    time_limit_sec: int = 300,
    solver_msg: bool = False,
) -> tuple:
    """Solve weighted p-median MILP for a single cluster.

    Returns
    -------
    (df_assignments, df_openings, metrics_dict, solver_status_str)
    """
    I_c = df_I[df_I["cluster_i"] == cluster_id].copy()

    _empty_assign = pd.DataFrame(columns=[
        "i", "p", "dist_m", "uncovered", "cluster_solved",
        "lat_i", "lon_i", "cluster_i", "w_i", "lat_p", "lon_p", "cluster_p",
    ])
    _empty_open = pd.DataFrame(columns=["id", "lat", "lon", "cluster", "opened", "cluster_solved"])
    _empty_met  = {
        "cluster": cluster_id, "opened": 0, "n_demands": 0,
        "covered": 0, "pct_covered": np.nan,
        "mean_dist_m": np.nan, "w_mean_dist_m": np.nan,
        "objective_value": np.nan,
    }

    if len(I_c) == 0:
        return _empty_assign, _empty_open, _empty_met, "NO_DEMAND"

    existing_all = df_P[df_P["es_existente"]].copy()
    cand_local   = df_P[(~df_P["es_existente"]) & (df_P["cluster_p"] == cluster_id)].copy()
    P_local_set  = set(pd.concat([existing_all, cand_local])["p"].astype(int))

    A_c = df_A[df_A["i"].isin(I_c["i"]) & df_A["p"].isin(P_local_set)].copy()

    mdl = pulp.LpProblem(f"c{cluster_id}_pmedian", pulp.LpMinimize)

    x = {
        (int(r.i), int(r.p)): pulp.LpVariable(f"x_{int(r.i)}_{int(r.p)}", 0, 1, "Binary")
        for r in A_c.itertuples(index=False)
    }
    u = {int(i): pulp.LpVariable(f"u_{int(i)}", 0, 1, "Binary") for i in I_c["i"].astype(int)}
    y = {int(p): pulp.LpVariable(f"y_{int(p)}", 0, 1, "Binary") for p in cand_local["p"].astype(int)}

    w_map     = dict(zip(I_c["i"].astype(int), I_c["w"].astype(float)))
    arcs_by_i = defaultdict(list)
    dist_map  = {}
    for r in A_c.itertuples(index=False):
        arcs_by_i[int(r.i)].append(int(r.p))
        dist_map[(int(r.i), int(r.p))] = float(r.dist_m)

    mdl += (
        pulp.lpSum(w_map[i] * dist_map[(i, p)] * x[(i, p)] for (i, p) in x)
        + pulp.lpSum(penalty_uncovered * w_map[i] * u[i] for i in u)
    )

    for i_id in I_c["i"].astype(int):
        pl = arcs_by_i.get(i_id, [])
        if not pl:
            mdl += (u[i_id] == 1)
        else:
            mdl += (pulp.lpSum(x[(i_id, p)] for p in pl) + u[i_id] == 1)

    for (i_id, p_id), var in x.items():
        if p_id in y:
            mdl += (var <= y[p_id])

    if y:
        mdl += (pulp.lpSum(y.values()) == min(int(p_new_per_cluster), len(y)))

    if y and len(df_conflictos) > 0:
        for r in df_conflictos.itertuples(index=False):
            pj = j_to_p_map.get(int(r.j))
            pk = j_to_p_map.get(int(r.k))
            if pj in y and pk in y:
                mdl += (y[pj] + y[pk] <= 1)

    pulp.PULP_CBC_CMD(msg=int(solver_msg), timeLimit=int(time_limit_sec)).solve(mdl)
    status = pulp.LpStatus.get(mdl.status, str(mdl.status))

    p_lkp = df_P.set_index("p")[["node_id", "lat", "lon", "cluster_p"]].to_dict("index")
    i_lkp = I_c.set_index("i")[["lat", "lon", "cluster_i", "w"]].to_dict("index")

    assign_rows = []
    for i_id, i_info in i_lkp.items():
        ap, ad = None, np.nan
        for p_id in arcs_by_i.get(int(i_id), []):
            v = x.get((int(i_id), int(p_id)))
            if v is not None and pulp.value(v) is not None and pulp.value(v) > 0.5:
                ap, ad = int(p_id), float(dist_map[(int(i_id), int(p_id))])
                break
        p_info = p_lkp.get(ap, {}) if ap is not None else {}
        assign_rows.append({
            "i":              int(i_id),
            "p":              float(ap) if ap is not None else np.nan,
            "dist_m":         ad,
            "uncovered":      0 if ap is not None else 1,
            "cluster_solved": cluster_id,
            "lat_i":          i_info["lat"],
            "lon_i":          i_info["lon"],
            "cluster_i":      i_info["cluster_i"],
            "w_i":            i_info["w"],
            "lat_p":          p_info.get("lat", np.nan),
            "lon_p":          p_info.get("lon", np.nan),
            "cluster_p":      p_info.get("cluster_p", np.nan),
        })
    df_assign = pd.DataFrame(assign_rows)

    opened_cands = [
        p_id for p_id, var in y.items()
        if pulp.value(var) is not None and pulp.value(var) > 0.5
    ]
    open_rows = [
        {
            "id":             p_id,
            "lat":            p_lkp.get(p_id, {}).get("lat", np.nan),
            "lon":            p_lkp.get(p_id, {}).get("lon", np.nan),
            "cluster":        cluster_id,
            "opened":         True,
            "cluster_solved": cluster_id,
        }
        for p_id in opened_cands
    ]
    df_open = pd.DataFrame(open_rows) if open_rows else _empty_open.copy()

    n_cov   = int((df_assign["uncovered"] == 0).sum())
    n_dem   = len(df_assign)
    cov_d   = df_assign.loc[df_assign["uncovered"] == 0, "dist_m"]
    cov_w   = df_assign.loc[df_assign["uncovered"] == 0, "w_i"]
    w_sum   = float(cov_w.sum())

    metrics = {
        "cluster":         cluster_id,
        "opened":          len(opened_cands),
        "n_demands":       n_dem,
        "covered":         n_cov,
        "pct_covered":     round(100.0 * n_cov / n_dem, 2) if n_dem > 0 else np.nan,
        "mean_dist_m":     round(float(cov_d.mean()), 2)           if len(cov_d) > 0 else np.nan,
        "w_mean_dist_m":   round(float((cov_d * cov_w).sum() / w_sum), 2) if w_sum > 0 else np.nan,
        "objective_value": pulp.value(mdl.objective),
    }
    return df_assign, df_open, metrics, status


# ─── MILP solver — global (no clustering) ────────────────────────────────────

def solve_global_pmedian(
    df_I: pd.DataFrame,
    df_J: pd.DataFrame,
    df_P: pd.DataFrame,
    df_A: pd.DataFrame,
    df_conflictos: pd.DataFrame,
    j_to_p_map: dict,
    p_total: int,
    penalty_uncovered: float = 5_000.0,
    time_limit_sec: int = 300,
    solver_msg: bool = False,
) -> tuple:
    """Solve global p-median (all demand in one cluster) as a single MILP.

    Returns
    -------
    (df_openings, metrics_dict, solver_status_str, runtime_s)
    """
    # Redirect through solve_cluster_model by putting everything in cluster 0
    _df_I = df_I.copy();  _df_I["cluster_i"]  = 0
    _df_P = df_P.copy();  _df_P.loc[~_df_P["es_existente"], "cluster_p"] = 0
    _df_J = _df_P[~_df_P["es_existente"]].copy().reset_index(drop=True)
    _df_J["j"] = range(len(_df_J))
    _j2p  = dict(zip(_df_J["j"].astype(int), _df_J["p"].astype(int)))

    t0 = time.time()
    _, df_open, metrics, status = solve_cluster_model(
        cluster_id=0,
        df_I=_df_I, df_J=_df_J, df_P=_df_P,
        df_A=df_A, df_conflictos=df_conflictos, j_to_p_map=_j2p,
        p_new_per_cluster=p_total,
        penalty_uncovered=penalty_uncovered,
        time_limit_sec=time_limit_sec,
        solver_msg=solver_msg,
    )
    runtime_s = round(time.time() - t0, 2)
    metrics["runtime_s"] = runtime_s

    # Attach graph_node to df_open
    p_to_gnode = dict(zip(df_P["p"].astype(int), df_P["graph_node"]))
    if len(df_open) > 0 and "id" in df_open.columns:
        df_open["graph_node"] = df_open["id"].astype(int).map(p_to_gnode)

    return df_open, metrics, status, runtime_s


# ─── Euclidean arc table builder ─────────────────────────────────────────────

def build_euclidean_arc_table(
    df_I: pd.DataFrame,
    df_P: pd.DataFrame,
    D_MAX: float,
    proj_epsg: int = 32614,
) -> pd.DataFrame:
    """Build arc table using projected Euclidean distances.

    Used for the B3 (Euclidean-solve) baseline.
    Requires df_I and df_P to have 'lat', 'lon' columns.

    Returns
    -------
    DataFrame with columns [i, p, dist_m] where dist_m is Euclidean (metres).
    """
    from shapely.geometry import Point

    def _project(df, epsg):
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        ).to_crs(epsg=epsg)
        return gdf.geometry.x.values, gdf.geometry.y.values

    xi, yi = _project(df_I, proj_epsg)
    xp, yp = _project(df_P, proj_epsg)

    i_ids = df_I["i"].astype(int).values
    p_ids = df_P["p"].astype(int).values

    rows = []
    for ii, (x0, y0) in enumerate(zip(xi, yi)):
        d = np.sqrt((xp - x0) ** 2 + (yp - y0) ** 2)
        for pi in np.where(d <= D_MAX)[0]:
            rows.append((int(i_ids[ii]), int(p_ids[pi]), float(d[pi])))

    df_arc_euc = pd.DataFrame(rows, columns=["i", "p", "dist_m"])
    if len(df_arc_euc) > 0:
        df_arc_euc = df_arc_euc.groupby(["i", "p"], as_index=False)["dist_m"].min()
    return df_arc_euc.reset_index(drop=True)


# ─── Network evaluation (uniform for ALL models) ─────────────────────────────

def evaluate_solution_network_distance(
    opened_gnodes,
    demand_df: pd.DataFrame,
    existing_gnodes,
    G_proj,
    D_MAX: float,
    model_name: str = "",
    n_openings=None,
    runtime_s=np.nan,
    solver_status: str = "OK",
    allocation_rule: str = "",
    solve_distance: str = "network",
    clustering: str = "",
    mca_used: str = "",
) -> dict:
    """Evaluate any facility solution using pedestrian-network distances.

    This is the single evaluation function used for every model so comparisons
    are always apples-to-apples.

    Parameters
    ----------
    opened_gnodes   : iterable of int — OSMnx node IDs for newly opened stores
    demand_df       : DataFrame with columns [graph_node (Int64), POBTOT, w]
    existing_gnodes : iterable of int — OSMnx node IDs for existing stores
    G_proj          : projected NetworkX DiGraph with 'length' edge attribute
    D_MAX           : float — max service radius in metres

    Returns
    -------
    dict with all fair-comparison-table fields.
    """
    t0 = time.time()

    _valid   = set(G_proj.nodes)
    _opened  = [int(n) for n in opened_gnodes  if pd.notna(n) and int(n) in _valid]
    _exist   = [int(n) for n in existing_gnodes if pd.notna(n) and int(n) in _valid]
    all_svc  = set(_opened) | set(_exist)

    if not all_svc:
        return dict(
            model=model_name, allocation_rule=allocation_rule,
            solve_distance=solve_distance, eval_distance="network",
            clustering=clustering, mca_used=mca_used,
            coverage_rate=np.nan, w_coverage_rate=np.nan,
            mean_dist_m=np.nan, w_mean_dist_m=np.nan,
            n_covered=0, n_demands=len(demand_df), n_openings=0,
            runtime_s=np.nan, eval_runtime_s=0.0,
            solver_status="NO_SERVICE_NODES",
        )

    dist_map = dict(
        nx.multi_source_dijkstra_path_length(
            G_proj, sources=all_svc, weight="length", cutoff=D_MAX * 5
        )
    )

    dd = demand_df.copy()
    dd["graph_node"] = dd["graph_node"].astype("Int64")
    dd["dist"]    = dd["graph_node"].map(
        lambda n: dist_map.get(int(n), np.inf) if pd.notna(n) else np.inf
    )
    dd["covered"] = dd["dist"] <= D_MAX

    n_dem    = len(dd)
    n_cov    = int(dd["covered"].sum())
    cov_r    = n_cov / n_dem if n_dem > 0 else np.nan

    w_tot    = float(dd["w"].sum())
    w_cov    = float(dd.loc[dd["covered"], "w"].sum())
    w_cov_r  = w_cov / w_tot if w_tot > 0 else np.nan

    cov_dd   = dd[dd["covered"]].copy()
    mean_d   = float(cov_dd["dist"].mean())          if len(cov_dd) > 0 else np.nan
    ws       = float(cov_dd["w"].sum())
    w_mean_d = float((cov_dd["dist"] * cov_dd["w"]).sum() / ws) if ws > 0 else np.nan

    eval_rt = round(time.time() - t0, 2)

    def _r(v, digits=4):
        return round(float(v), digits) if (v is not None and not np.isnan(float(v))) else np.nan

    return dict(
        model           = model_name,
        allocation_rule = allocation_rule,
        solve_distance  = solve_distance,
        eval_distance   = "network",
        clustering      = clustering,
        mca_used        = mca_used,
        coverage_rate   = _r(cov_r),
        w_coverage_rate = _r(w_cov_r),
        mean_dist_m     = _r(mean_d, 2),
        w_mean_dist_m   = _r(w_mean_d, 2),
        n_covered       = n_cov,
        n_demands       = n_dem,
        n_openings      = n_openings if n_openings is not None else len(_opened),
        runtime_s       = runtime_s,
        eval_runtime_s  = eval_rt,
        solver_status   = solver_status,
        coverage_pct    = _r(cov_r * 100, 2),
        w_coverage_pct  = _r(w_cov_r * 100, 2),
    )


# ─── High-level model runners ─────────────────────────────────────────────────

def run_proposed_model(
    df_I: pd.DataFrame,
    df_J: pd.DataFrame,
    df_P: pd.DataFrame,
    df_A: pd.DataFrame,
    df_conflictos: pd.DataFrame,
    j_to_p_map: dict,
    cluster_ids: list,
    alloc: dict,
    G_proj,
    demand_eval: pd.DataFrame,
    existing_gnodes,
    D_MAX: float,
    penalty_uncovered: float = 5_000.0,
    time_limit_sec: int = 300,
    model_name: str = "Proposed",
    allocation_rule: str = "",
    solve_distance: str = "network",
    clustering: str = "spectral",
    mca_used: str = "yes",
    verbose: bool = True,
) -> tuple:
    """Run the full cluster-based p-median pipeline.

    Returns
    -------
    (df_openings, df_assignments, df_metrics, eval_result_dict, total_runtime_s)
    """
    all_open, all_assign, all_metrics = [], [], []
    p_to_gnode = dict(zip(df_P["p"].astype(int), df_P["graph_node"]))

    t0 = time.time()
    for c in cluster_ids:
        if verbose:
            print(f"  Cluster {c}  (p_new={alloc.get(c, '?')}) …", end=" ")
        da, do, dm, ds = solve_cluster_model(
            cluster_id=c,
            df_I=df_I, df_J=df_J, df_P=df_P, df_A=df_A,
            df_conflictos=df_conflictos, j_to_p_map=j_to_p_map,
            p_new_per_cluster=alloc.get(c, 7),
            penalty_uncovered=penalty_uncovered,
            time_limit_sec=time_limit_sec,
        )
        if verbose:
            print(f"status={ds}  opened={dm['opened']}  covered={dm['covered']}/{dm['n_demands']}")
        all_open.append(do)
        all_assign.append(da)
        all_metrics.append({**dm, "solver_status": ds})

    runtime_s = round(time.time() - t0, 2)
    df_open   = pd.concat(all_open,   ignore_index=True) if all_open   else pd.DataFrame()
    df_assign = pd.concat(all_assign, ignore_index=True) if all_assign else pd.DataFrame()
    df_met    = pd.DataFrame(all_metrics)

    opened_gnodes = []
    if len(df_open) > 0 and "id" in df_open.columns:
        for p_id in df_open["id"].astype(int):
            gn = p_to_gnode.get(p_id)
            if pd.notna(gn):
                opened_gnodes.append(int(gn))

    eval_result = evaluate_solution_network_distance(
        opened_gnodes=opened_gnodes,
        demand_df=demand_eval,
        existing_gnodes=existing_gnodes,
        G_proj=G_proj,
        D_MAX=D_MAX,
        model_name=model_name,
        n_openings=len(df_open),
        runtime_s=runtime_s,
        solver_status="OK",
        allocation_rule=allocation_rule,
        solve_distance=solve_distance,
        clustering=clustering,
        mca_used=mca_used,
    )
    return df_open, df_assign, df_met, eval_result, runtime_s


def run_baseline_models(
    df_I: pd.DataFrame,
    df_J: pd.DataFrame,
    df_P: pd.DataFrame,
    df_A: pd.DataFrame,
    df_conflictos: pd.DataFrame,
    j_to_p_map: dict,
    data_condensed: pd.DataFrame,
    cluster_ids: list,
    fixed_alloc: dict,
    G_proj,
    demand_eval: pd.DataFrame,
    existing_gnodes,
    D_MAX: float,
    P_TOTAL: int,
    RANDOM_SEED: int,
    PROJ_EPSG: int,
    penalty_uncovered: float = 5_000.0,
    time_limit_sec: int = 300,
    verbose: bool = True,
) -> dict:
    """Run B1, B2, B3, B4 baselines and return evaluation dicts.

    Returns
    -------
    dict with keys 'B1', 'B2', 'B3', 'B4', each holding an eval_result dict.
    Also 'B2_open', 'B3_open' DataFrames and 'B4_open'.
    """
    import geopandas as gpd
    from sklearn.cluster import KMeans

    n_E = len(df_P[df_P["es_existente"]])
    p_to_gnode = dict(zip(df_P["p"].astype(int), df_P["graph_node"]))

    results = {}

    # ── B1: Global P-Median ──────────────────────────────────────────────────
    if verbose:
        print("B1: Global P-Median (no clustering, p={P_TOTAL}) …".replace("{P_TOTAL}", str(P_TOTAL)))
    _df_I_b1 = df_I.copy();  _df_I_b1["cluster_i"]  = 0
    _df_P_b1 = df_P.copy();  _df_P_b1.loc[~_df_P_b1["es_existente"], "cluster_p"] = 0
    _df_J_b1 = _df_P_b1[~_df_P_b1["es_existente"]].copy().reset_index(drop=True)
    _df_J_b1["j"] = range(len(_df_J_b1))
    _j2p_b1 = dict(zip(_df_J_b1["j"].astype(int), _df_J_b1["p"].astype(int)))

    t0 = time.time()
    _, _open_b1, _met_b1, _stat_b1 = solve_cluster_model(
        cluster_id=0,
        df_I=_df_I_b1, df_J=_df_J_b1, df_P=_df_P_b1,
        df_A=df_A, df_conflictos=df_conflictos, j_to_p_map=_j2p_b1,
        p_new_per_cluster=P_TOTAL,
        penalty_uncovered=penalty_uncovered, time_limit_sec=time_limit_sec,
    )
    _t_b1 = round(time.time() - t0, 2)
    if verbose:
        print(f"  status={_stat_b1}  opened={_met_b1['opened']}  t={_t_b1}s")

    _b1_gnodes = []
    for _, _r in _open_b1.iterrows():
        if _r.get("opened", True):
            gn = p_to_gnode.get(int(_r["id"]))
            if pd.notna(gn):
                _b1_gnodes.append(int(gn))

    results["B1"] = evaluate_solution_network_distance(
        _b1_gnodes, demand_eval, existing_gnodes, G_proj, D_MAX,
        model_name="B1: Global P-Median (no clustering)",
        n_openings=len(_b1_gnodes), runtime_s=_t_b1, solver_status=_stat_b1,
        allocation_rule=f"global p={P_TOTAL}", solve_distance="network",
        clustering="none", mca_used="yes",
    )
    results["B1_open"] = _open_b1

    # ── B2: K-Means + P-Median ───────────────────────────────────────────────
    if verbose:
        print(f"B2: K-Means (k={len(cluster_ids)}) + P-Median …")
    _n_k = len(cluster_ids)

    _gdf_all = gpd.GeoDataFrame(
        data_condensed,
        geometry=gpd.points_from_xy(data_condensed["lon"], data_condensed["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=PROJ_EPSG)
    _all_coords = np.column_stack([_gdf_all.geometry.x.values, _gdf_all.geometry.y.values])

    _km = KMeans(n_clusters=_n_k, random_state=RANDOM_SEED, n_init=20)
    _km_labels = _km.fit_predict(_all_coords)
    _km_series = pd.Series(_km_labels)
    _km_map    = {old: new for new, old in enumerate(_km_series.value_counts().sort_values(ascending=False).index)}
    _km_labels = _km_series.map(_km_map).values

    _nid_to_km = {int(nid): int(_km_labels[i]) for i, nid in enumerate(data_condensed["node_id"])}

    _df_I_b2 = df_I.copy()
    _df_I_b2["cluster_i"] = _df_I_b2["node_id"].map(_nid_to_km).fillna(0).astype(int)

    _df_P_b2 = df_P.copy()
    _cand_m = ~_df_P_b2["es_existente"]
    _df_P_b2.loc[_cand_m, "cluster_p"] = _df_P_b2.loc[_cand_m, "node_id"].map(_nid_to_km).fillna(0).astype(int)
    _df_J_b2 = _df_P_b2[_cand_m].copy().reset_index(drop=True)
    _df_J_b2["j"] = (_df_J_b2["p"].astype(int) - n_E)

    _clusters_b2 = sorted(_df_I_b2["cluster_i"].dropna().astype(int).unique().tolist())
    _pnew_b2 = compute_pnew_per_cluster(_df_I_b2, _clusters_b2, "fixed", 7, P_TOTAL)

    t0 = time.time()
    _all_open_b2 = []
    for _c2 in _clusters_b2:
        _, _o2, _m2, _s2 = solve_cluster_model(
            cluster_id=_c2,
            df_I=_df_I_b2, df_J=_df_J_b2, df_P=_df_P_b2,
            df_A=df_A, df_conflictos=df_conflictos, j_to_p_map=j_to_p_map,
            p_new_per_cluster=_pnew_b2.get(_c2, 7),
            penalty_uncovered=penalty_uncovered, time_limit_sec=time_limit_sec,
        )
        _all_open_b2.append(_o2)
        if verbose:
            print(f"    C{_c2}: {_s2}  opened={_m2['opened']}  covered={_m2['covered']}/{_m2['n_demands']}")
    _t_b2 = round(time.time() - t0, 2)

    _open_b2 = pd.concat(_all_open_b2, ignore_index=True) if _all_open_b2 else pd.DataFrame()
    _b2_gnodes = []
    for _, _r in _open_b2.iterrows():
        if _r.get("opened", True):
            gn = p_to_gnode.get(int(_r["id"]))
            if pd.notna(gn):
                _b2_gnodes.append(int(gn))

    results["B2"] = evaluate_solution_network_distance(
        _b2_gnodes, demand_eval, existing_gnodes, G_proj, D_MAX,
        model_name="B2: K-Means + P-Median",
        n_openings=len(_b2_gnodes), runtime_s=_t_b2, solver_status="OK",
        allocation_rule="fixed 7/cluster", solve_distance="network",
        clustering="k-means", mca_used="yes",
    )
    results["B2_open"] = _open_b2

    # ── B3: Euclidean solve, network evaluation ───────────────────────────────
    if verbose:
        print("B3: Euclidean-distance solve → network evaluation …")
    df_A_euc = build_euclidean_arc_table(df_I, df_P, D_MAX, proj_epsg=PROJ_EPSG)

    t0 = time.time()
    _all_open_b3 = []
    for _c3 in cluster_ids:
        _, _o3, _m3, _s3 = solve_cluster_model(
            cluster_id=_c3,
            df_I=df_I, df_J=df_J, df_P=df_P,
            df_A=df_A_euc, df_conflictos=df_conflictos, j_to_p_map=j_to_p_map,
            p_new_per_cluster=fixed_alloc.get(_c3, 7),
            penalty_uncovered=penalty_uncovered, time_limit_sec=time_limit_sec,
        )
        _all_open_b3.append(_o3)
        if verbose:
            print(f"    C{_c3}: {_s3}  opened={_m3['opened']}")
    _t_b3 = round(time.time() - t0, 2)

    _open_b3 = pd.concat(_all_open_b3, ignore_index=True) if _all_open_b3 else pd.DataFrame()
    _b3_gnodes = []
    for _, _r in _open_b3.iterrows():
        if _r.get("opened", True):
            gn = p_to_gnode.get(int(_r["id"]))
            if pd.notna(gn):
                _b3_gnodes.append(int(gn))

    results["B3"] = evaluate_solution_network_distance(
        _b3_gnodes, demand_eval, existing_gnodes, G_proj, D_MAX,
        model_name="B3: Euclidean solve, network evaluation",
        n_openings=len(_b3_gnodes), runtime_s=_t_b3, solver_status="OK",
        allocation_rule="fixed 7/cluster",
        solve_distance="Euclidean (corrected eval=network)",
        clustering="spectral", mca_used="yes",
    )
    results["B3_open"] = _open_b3

    # ── B4: No MCA (β = 0, population-only weights) ──────────────────────────
    if verbose:
        print("B4: No MCA (β=0, population weights) …")
    _df_I_b4 = df_I.copy()
    _df_I_b4["w"] = _df_I_b4["POBTOT"].astype(float).clip(lower=0.0)

    t0 = time.time()
    _all_open_b4 = []
    for _c4 in cluster_ids:
        _, _o4, _m4, _s4 = solve_cluster_model(
            cluster_id=_c4,
            df_I=_df_I_b4, df_J=df_J, df_P=df_P,
            df_A=df_A, df_conflictos=df_conflictos, j_to_p_map=j_to_p_map,
            p_new_per_cluster=fixed_alloc.get(_c4, 7),
            penalty_uncovered=penalty_uncovered, time_limit_sec=time_limit_sec,
        )
        _all_open_b4.append(_o4)
        if verbose:
            print(f"    C{_c4}: {_s4}  opened={_m4['opened']}")
    _t_b4 = round(time.time() - t0, 2)

    _open_b4 = pd.concat(_all_open_b4, ignore_index=True) if _all_open_b4 else pd.DataFrame()
    _b4_gnodes = []
    for _, _r in _open_b4.iterrows():
        if _r.get("opened", True):
            gn = p_to_gnode.get(int(_r["id"]))
            if pd.notna(gn):
                _b4_gnodes.append(int(gn))

    results["B4"] = evaluate_solution_network_distance(
        _b4_gnodes, demand_eval, existing_gnodes, G_proj, D_MAX,
        model_name="B4: No MCA (β=0)",
        n_openings=len(_b4_gnodes), runtime_s=_t_b4, solver_status="OK",
        allocation_rule="fixed 7/cluster", solve_distance="network",
        clustering="spectral", mca_used="no (β=0)",
    )
    results["B4_open"] = _open_b4

    return results


# ─── Grid search ──────────────────────────────────────────────────────────────

def run_parameter_grid(
    df_I: pd.DataFrame,
    df_J: pd.DataFrame,
    df_P: pd.DataFrame,
    df_A: pd.DataFrame,
    df_conflictos: pd.DataFrame,
    j_to_p_map: dict,
    cluster_ids: list,
    G_proj,
    demand_eval: pd.DataFrame,
    existing_gnodes,
    d_max_grid: list,
    s_min_grid: list,
    beta_grid: list,
    p_total_grid: list,
    penalty_uncovered: float = 5_000.0,
    time_limit_sec: int = 300,
    mode: str = "demand_weighted",
    verbose: bool = True,
) -> pd.DataFrame:
    """Grid search over (D_MAX, S_MIN, beta, P_total).

    Arc filtering: df_A is filtered to dist_m ≤ D_MAX (approximation that avoids
    rebuilding the full shortest-path table per D_MAX value).

    Returns
    -------
    DataFrame with one row per (D_MAX, S_MIN, beta, P_total) combination.
    """
    from itertools import product

    combos = list(product(d_max_grid, s_min_grid, beta_grid, p_total_grid))
    if verbose:
        print(f"Grid search: {len(combos)} combinations")

    rows = []
    for idx, (d_max, s_min, beta, p_total) in enumerate(combos):
        if verbose:
            print(f"  [{idx+1}/{len(combos)}] D_MAX={d_max}, S_MIN={s_min}, β={beta}, P={p_total}", end=" … ")

        df_A_cur = df_A[df_A["dist_m"] <= d_max].copy()
        df_C_cur = df_conflictos[df_conflictos["dist_m"] <= s_min].copy() if len(df_conflictos) > 0 else df_conflictos

        # Re-weight demand for this beta
        _pop   = df_I["POBTOT"].astype(float)
        _mca1  = df_I["MCA1"].astype(float)
        _std   = float(_mca1.std(ddof=0)) if len(_mca1) > 1 else 1.0
        _z     = (_mca1 - float(_mca1.mean())) / (_std if _std > 0 else 1.0)
        df_I_b = df_I.copy()
        df_I_b["w"] = (_pop * (1.0 + beta * _z.fillna(0.0))).clip(lower=0.0)

        alloc  = compute_pnew_per_cluster(df_I_b, cluster_ids, mode, 7, p_total)

        t0 = time.time()
        all_open = []
        status_ok = True
        for c in cluster_ids:
            _, do, dm, ds = solve_cluster_model(
                cluster_id=c,
                df_I=df_I_b, df_J=df_J, df_P=df_P,
                df_A=df_A_cur, df_conflictos=df_C_cur, j_to_p_map=j_to_p_map,
                p_new_per_cluster=alloc.get(c, 7),
                penalty_uncovered=penalty_uncovered,
                time_limit_sec=time_limit_sec,
            )
            all_open.append(do)
            if ds not in ("Optimal", "OK"):
                status_ok = False
        runtime_s = round(time.time() - t0, 2)

        p_to_gnode = dict(zip(df_P["p"].astype(int), df_P["graph_node"]))
        df_open_all = pd.concat(all_open, ignore_index=True) if all_open else pd.DataFrame()
        opened_gnodes = []
        if len(df_open_all) > 0 and "id" in df_open_all.columns:
            for p_id in df_open_all["id"].astype(int):
                gn = p_to_gnode.get(p_id)
                if pd.notna(gn):
                    opened_gnodes.append(int(gn))

        ev = evaluate_solution_network_distance(
            opened_gnodes=opened_gnodes,
            demand_df=demand_eval,
            existing_gnodes=existing_gnodes,
            G_proj=G_proj,
            D_MAX=d_max,
            runtime_s=runtime_s,
            solver_status="OK" if status_ok else "SUBOPTIMAL",
        )

        rows.append({
            "D_MAX":           d_max,
            "S_MIN":           s_min,
            "beta":            beta,
            "p_total":         p_total,
            "coverage_rate":   ev["coverage_rate"],
            "w_coverage_rate": ev["w_coverage_rate"],
            "mean_dist_m":     ev["mean_dist_m"],
            "w_mean_dist_m":   ev["w_mean_dist_m"],
            "n_covered":       ev["n_covered"],
            "n_demands":       ev["n_demands"],
            "n_openings":      ev["n_openings"],
            "runtime_s":       runtime_s,
            "solver_status":   ev["solver_status"],
        })

        if verbose:
            cov = ev.get("coverage_rate")
            print(f"cov={cov:.3f}  runtime={runtime_s}s" if cov is not None else f"runtime={runtime_s}s")

    return pd.DataFrame(rows)


# ─── Pareto frontier ──────────────────────────────────────────────────────────

def compute_pareto_frontier(
    df: pd.DataFrame,
    maximize_cols: list = None,
    minimize_cols: list = None,
) -> pd.DataFrame:
    """Add 'is_pareto' column. A row is Pareto-efficient if nothing dominates it.

    Parameters
    ----------
    maximize_cols : list of column names where higher is better
    minimize_cols : list of column names where lower is better
    """
    maximize_cols = maximize_cols or []
    minimize_cols = minimize_cols or []
    obj_cols = maximize_cols + minimize_cols

    df_out = df.copy()
    df_out["is_pareto"] = False

    df_clean = df.dropna(subset=obj_cols).copy()
    if len(df_clean) == 0:
        return df_out

    orig_idx = df_clean.index.tolist()

    # Convert all objectives to maximization
    n_max = len(maximize_cols)
    vals  = np.zeros((len(df_clean), len(obj_cols)))
    for j, col in enumerate(maximize_cols):
        vals[:, j] = df_clean[col].values
    for j, col in enumerate(minimize_cols):
        vals[:, n_max + j] = -df_clean[col].values

    n = len(vals)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for k in range(n):
            if i == k:
                continue
            if np.all(vals[k] >= vals[i]) and np.any(vals[k] > vals[i]):
                dominated[i] = True
                break

    for i, idx in enumerate(orig_idx):
        df_out.at[idx, "is_pareto"] = not bool(dominated[i])
    return df_out


def select_calibrated_params(
    df_grid: pd.DataFrame,
    p_total_fixed: int = 42,
    weights: dict = None,
) -> dict:
    """Select the highest-scoring Pareto-optimal configuration.

    Scoring rule (default):
        score = 0.40 × norm_w_coverage + 0.30 × norm_coverage
              - 0.20 × norm_w_distance  - 0.10 × norm_runtime

    Returns dict with selected parameters and score.
    """
    weights = weights or {
        "w_coverage_rate": 0.40,
        "coverage_rate":   0.30,
        "w_mean_dist_m":  -0.20,
        "runtime_s":      -0.10,
    }

    _g = df_grid[
        (df_grid["p_total"] == p_total_fixed) &
        (df_grid["solver_status"].isin(["OK", "Optimal"]))
    ].copy().reset_index(drop=True)

    if len(_g) == 0:
        _g = df_grid[df_grid["solver_status"].isin(["OK", "Optimal"])].copy().reset_index(drop=True)

    if len(_g) == 0:
        return {}

    _g = compute_pareto_frontier(
        _g,
        maximize_cols=["coverage_rate", "w_coverage_rate"],
        minimize_cols=["w_mean_dist_m", "runtime_s"],
    )
    pareto = _g[_g["is_pareto"]].copy()
    if len(pareto) == 0:
        pareto = _g.copy()

    # Normalize objectives
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    pareto = pareto.copy()
    pareto["score"] = 0.0
    for col, w in weights.items():
        if col in pareto.columns:
            normed = _norm(pareto[col].astype(float))
            if w < 0:
                normed = 1.0 - normed
            pareto["score"] += abs(w) * normed

    best = pareto.loc[pareto["score"].idxmax()]

    return {
        "D_MAX":           float(best.get("D_MAX", np.nan)),
        "S_MIN":           float(best.get("S_MIN", np.nan)),
        "beta":            float(best.get("beta", np.nan)),
        "p_total":         int(best.get("p_total", p_total_fixed)),
        "coverage_rate":   float(best.get("coverage_rate", np.nan)),
        "w_coverage_rate": float(best.get("w_coverage_rate", np.nan)),
        "w_mean_dist_m":   float(best.get("w_mean_dist_m", np.nan)),
        "runtime_s":       float(best.get("runtime_s", np.nan)),
        "score":           float(best["score"]),
        "n_pareto_points": int(len(pareto)),
    }


# ─── Parameter justification summary ─────────────────────────────────────────

def summarize_parameter_justification(
    df_grid: pd.DataFrame,
    chosen_params: dict,
    avg_nn_m: float,
    sens_summary: dict,
) -> pd.DataFrame:
    """Build a parameter justification table suitable for paper appendix."""
    D_MAX = chosen_params.get("D_MAX", 366.0)
    S_MIN = chosen_params.get("S_MIN", 240.0)
    BETA  = chosen_params.get("beta", 0.25)

    rows = []
    if len(df_grid) > 0 and "coverage_rate" in df_grid.columns:
        best_cov = df_grid.loc[df_grid["coverage_rate"].idxmax()]
        rows.append({
            "Parameter": "D_MAX (m)", "Chosen": D_MAX,
            "Justification":
                f"~5-min walk at 1.22 m/s; best grid-search coverage at D_MAX={int(best_cov.get('D_MAX', D_MAX))}",
        })
    else:
        rows.append({"Parameter": "D_MAX (m)", "Chosen": D_MAX,
                     "Justification": "~5-min walk at 1.22 m/s"})

    rows.append({
        "Parameter": "S_MIN (m)", "Chosen": S_MIN,
        "Justification": "Minimum candidate spacing; prevents clustering of new openings",
    })
    rows.append({
        "Parameter": "β (MCA weight)", "Chosen": BETA,
        "Justification": f"β={BETA} preserves pop-MCA correlation; sensitivity sweep confirms stability",
    })
    rows.append({
        "Parameter": "K (KNN affinity)", "Chosen": sens_summary.get("knn_k_chosen", 12),
        "Justification": "Eigengap-selected k*; validated by silhouette score in spectral embedding",
    })
    rows.append({
        "Parameter": "Threshold multiplier", "Chosen": sens_summary.get("threshold_multiplier_chosen", 1.0),
        "Justification": f"1.0 × avg_nn_m = {avg_nn_m:.1f} m mirrors empirical existing-store spacing",
    })
    rows.append({
        "Parameter": "P_total", "Chosen": chosen_params.get("p_total", 42),
        "Justification": "Budget: 42 new openings (comparable across all models)",
    })
    return pd.DataFrame(rows)


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def save_table(df: pd.DataFrame, path, name: str = "") -> None:
    """Save DataFrame to CSV with console confirmation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved table  → {path}  ({len(df)} rows × {len(df.columns)} cols)")


def save_figure(fig, path, name: str = "", dpi: int = 150) -> None:
    """Save matplotlib figure with console confirmation and close it."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved figure → {path}")
    plt.close(fig)


def build_pmedian_inputs_graph(*args, **kwargs):
    """Placeholder: build p-median inputs from raw data using graph shortest paths.

    This is done in f_l_notebook_revised.ipynb (the preprocessing notebook).
    The publication notebook loads the pre-computed bundle instead.
    """
    raise NotImplementedError(
        "build_pmedian_inputs_graph is handled by the preprocessing notebook.\n"
        "Run f_l_notebook_revised.ipynb to generate the pmedian_bundle files."
    )
