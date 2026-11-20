"""Open-set CV splits that mimic the test split. The ONLY module that decides gallery vs query membership.

Per species and seed:
  1. Encounter clusters: inside each identity, average-linkage on cosine distance with cut `enc_thr`.
     The test set is encounter-disjoint from train (EDA §4), so whole clusters move together.
  2. "New" identities: a fraction `f_new_id` of identities (with <= max_new_id_size images), stratified by shot bin,
     are removed from the gallery entirely; all their images become queries labelled new_individual.
  3. Known queries: for every other identity with >= 2 clusters, each cluster moves to the query set with
     probability `p_move`; at least one cluster always stays in the gallery.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import NEW, SPECIES

SHOT_BINS = [0, 1, 2, 5, 10, 20, 10 ** 6]


@dataclass
class FoldData:
    """Everything a model may see when fitting/predicting on one fold. NO query labels."""
    species: str
    gallery_rows: np.ndarray          # rows into Data.meta / X
    gallery_labels: np.ndarray        # identity strings
    query_rows: np.ndarray
    query_ids: np.ndarray
    extra: dict = field(default_factory=dict)


@dataclass
class FoldTruth:
    query_ids: np.ndarray
    labels: np.ndarray                # identity or new_individual

    def as_dict(self):
        return dict(zip(self.query_ids, self.labels))


def l2n(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def encounter_clusters(U, labels, thr):
    """U: L2-normalised embeddings of one species' train images; returns cluster key per image."""
    out = np.empty(len(labels), dtype=object)
    for ident in np.unique(labels):
        idx = np.where(labels == ident)[0]
        if len(idx) == 1:
            out[idx] = f"{ident}#0"
            continue
        Z = U[idx]
        D = np.clip(1 - Z @ Z.T, 0, 2)
        np.fill_diagonal(D, 0)
        lab = fcluster(linkage(squareform(D, checks=False), "average"), thr, "distance")
        out[idx] = [f"{ident}#{k}" for k in lab]
    return out


def make_split(labels, clusters, seed, p_move, f_new_id, max_new_id_size, species):
    """Returns boolean query mask and truth labels (NEW for held-out identities). Indices are local to the species."""
    rng = np.random.default_rng(seed * 7919 + SPECIES.index(species))
    s = pd.Series(labels)
    size = s.value_counts()
    elig = size[size <= max_new_id_size]
    bins = pd.cut(elig, SHOT_BINS)
    new_ids = []
    for _, grp in elig.groupby(bins, observed=True):
        k = int(round(f_new_id * len(grp)))
        if k:
            new_ids += list(rng.choice(grp.index.values, k, replace=False))
    new_ids = set(new_ids)
    is_query = np.zeros(len(labels), bool)
    is_query[s.isin(new_ids).values] = True
    cl = pd.Series(clusters)
    for ident in size.index:
        if ident in new_ids:
            continue
        idx = np.where(labels == ident)[0]
        ucl = np.unique(cl.values[idx])
        if len(ucl) < 2:
            continue
        move = rng.random(len(ucl)) < p_move
        if move.all():
            move[rng.integers(len(ucl))] = False
        moved = set(ucl[move])
        is_query[idx[np.isin(cl.values[idx], list(moved))]] = True
    truth = np.where(s.isin(new_ids).values, NEW, labels)
    return is_query, truth


def build_splits(data, cfg, U=None):
    """Returns list over seeds of dict species -> (FoldData, FoldTruth). Cached as json of ids."""
    from .io import P
    c = cfg["cv"]
    key = json.dumps({k: c[k] for k in ("seeds", "enc_thr", "p_move", "f_new_id", "max_new_id_size")}, sort_keys=True)
    import hashlib
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    path = P(cfg, "cache") / "splits" / f"splits_{h}.json"
    tr = data.train
    if U is None:
        U = l2n(data.X.astype(np.float64))
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = {"config": json.loads(key), "splits": []}
        clus = {}
        for sp in SPECIES:
            m = (tr.species == sp).values
            thr = c["enc_thr"][sp] if isinstance(c["enc_thr"], dict) else c["enc_thr"]
            clus[sp] = encounter_clusters(U[tr.row.values[m]], tr.identity.values[m], thr)
        for seed in c["seeds"]:
            entry = {}
            for sp in SPECIES:
                m = (tr.species == sp).values
                rows = tr.row.values[m]; labels = tr.identity.values[m]
                cap = c["max_new_id_size"][sp] if isinstance(c["max_new_id_size"], dict) else c["max_new_id_size"]
                isq, truth = make_split(labels, clus[sp], seed, c["p_move"][sp], c["f_new_id"][sp], cap, sp)
                entry[sp] = {"gallery": rows[~isq].tolist(), "query": rows[isq].tolist(),
                             "truth": truth[isq].tolist()}
            raw["splits"].append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    out = []
    ident = data.meta.identity.values
    ids = data.meta.id.values
    for entry in raw["splits"]:
        e = {}
        for sp in SPECIES:
            g = np.array(entry[sp]["gallery"]); q = np.array(entry[sp]["query"])
            e[sp] = (FoldData(sp, g, ident[g], q, ids[q]), FoldTruth(ids[q], np.array(entry[sp]["truth"])))
        out.append(e)
    return out


def test_fold(data, sp):
    """Final inference fold: full training gallery of the species, test queries (no truth exists)."""
    tr, te = data.train, data.test
    g = tr.row.values[(tr.species == sp).values]
    q = te.row.values[(te.species == sp).values]
    return FoldData(sp, g, data.meta.identity.values[g], q, data.meta.id.values[q])
