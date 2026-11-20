"""Decision rule (sort labels by probability, output 5) and the ONLY submission writer (always validated)."""
import numpy as np
import pandas as pd

from . import NEW


def rank_labels(cands, probs, p_new, k=5):
    """cands: list of identity labels, probs: their probabilities, p_new: P(new). Returns top-k labels sorted by prob."""
    labels = list(cands) + [NEW]
    p = np.r_[np.asarray(probs, float), p_new]
    order = np.argsort(-p, kind="stable")
    return [labels[i] for i in order[:k]]


def validate(sub: pd.DataFrame, test_meta: pd.DataFrame, train_meta: pd.DataFrame):
    assert list(sub.columns) == ["id", "predictions"], sub.columns
    assert len(sub) == len(test_meta) == 2791, len(sub)
    assert set(sub.id) == set(test_meta.id) and not sub.id.duplicated().any()
    ids_by_sp = train_meta.groupby("species").identity.apply(set).to_dict()
    sp = test_meta.set_index("id").species
    for tid, pred in zip(sub.id, sub.predictions):
        labs = pred.split(" ")
        assert 1 <= len(labs) <= 5 and len(set(labs)) == len(labs), (tid, pred)
        for lab in labs:
            assert lab == NEW or lab in ids_by_sp[sp[tid]], (tid, lab)


def write_submission(preds: dict, test_meta, train_meta, path):
    sub = pd.DataFrame({"id": test_meta.id.values, "predictions": [" ".join(preds[i]) for i in test_meta.id]})
    validate(sub, test_meta, train_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
