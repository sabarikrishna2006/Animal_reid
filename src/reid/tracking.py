"""Experiment registry: experiments/<exp_id>/{config,metrics}.json + registry.csv row."""
import csv
import datetime as dt
import json

from .io import P

FIELDS = ["exp_id", "date", "parent_exp", "change", "cv_mean", "cv_std", "map_hyena", "map_leopard", "known_map",
          "new_auc", "ks", "accepted", "lb_public"]


def new_exp_id(slug):
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + slug


def save_experiment(cfg, exp_id, metrics, notes=""):
    d = P(cfg, "experiments") / exp_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if notes:
        (d / "notes.md").write_text(notes)
    return d


def append_registry(cfg, row: dict):
    path = P(cfg, "experiments") / "registry.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
