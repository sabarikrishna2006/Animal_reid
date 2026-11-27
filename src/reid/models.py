"""Second-level models: LightGBM reranker (query, identity) and P(new | query). Out-of-fold by query identity."""
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

NON_FEATURES = {"p", "p_lgb", "p_xgb", "p_cat", "p_lr", "p_mlp", "qi", "cand", "query_id", "cand_id", "y", "split", "group", "species", "is_new", "truth"}
MONO_POS = ("max_l2", "max_pcaw", "max_lda", "sift_inl_max", "sift_inl_strict_max")

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=10, seed=2026)


def feature_cols(df, drop=()):
    return [c for c in df.columns if c not in NON_FEATURES and c not in drop and df[c].dtype != object]


class LGBModel:
    def __init__(self, params=None, mono=True, max_rounds=3000):
        self.params = dict(LGB_PARAMS, **(params or {})); self.mono = mono; self.max_rounds = max_rounds

    def fit(self, X, y, groups, feats):
        p = dict(self.params)
        if self.mono:
            if "sift_inl_clean2" in feats:
                mono_set = ("max_l2", "max_pcaw", "max_lda", "sift_inl_clean2", "hub_l2", "hub_lda", "lnb_rimg",
                            "max_lda512", "max_slda512", "max_sw2152", "max_ssw2152",
                            "max_proto_sw2152", "max_dba_sw2152")
            elif "sift_inl_clean" in feats:
                mono_set = ("max_l2", "max_pcaw", "max_lda", "sift_inl_clean")
            else:
                mono_set = MONO_POS
            p["monotone_constraints"] = [1 if f in mono_set else 0 for f in feats]
            p["monotone_constraints_method"] = "advanced"
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, y, groups))
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=False)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        self.model = lgb.train(p, dtr, self.max_rounds, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iter = self.model.best_iteration
        return self

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.best_iter)


class LRModel:
    def __init__(self, C=0.3):
        self.C = C

    def fit(self, X, y, groups, feats):
        self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                   LogisticRegression(C=self.C, max_iter=3000))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBModel:
    def fit(self, X, y, groups, feats):
        import xgboost as xgb
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, y, groups))
        self.model = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8,
                                       colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=10,
                                       tree_method="hist", early_stopping_rounds=100, eval_metric="logloss",
                                       random_state=2026)
        self.model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        self.best_iter = self.model.best_iteration
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class CatModel:
    def fit(self, X, y, groups, feats):
        from catboost import CatBoostClassifier
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, y, groups))
        self.model = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                                        thread_count=10, random_seed=2026, verbose=False, od_type="Iter", od_wait=100)
        self.model.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        self.best_iter = self.model.get_best_iteration()
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class MLPModel:
    def fit(self, X, y, groups, feats):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import FunctionTransformer
        clip = FunctionTransformer(lambda a: np.clip(a, -6.0, 6.0))
        self.models = []
        for sd in (2026, 2027):
            m = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                clip,
                MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.05,
                              batch_size=512, learning_rate_init=1e-3, early_stopping=True,
                              validation_fraction=0.12, n_iter_no_change=6, max_iter=45, random_state=sd),
            )
            m.fit(X, y)
            self.models.append(m)
        return self

    def predict(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)


def make_model(kind, **kw):
    return {"lgb": LGBModel, "lr": LRModel, "xgb": XGBModel, "cat": CatModel, "mlp": MLPModel}[kind](**kw)


def oof_predict(df, feats, label, groups, kind="lgb", n_splits=5, return_models=False, seeds=None, **kw):
    """Out-of-fold predictions with GroupKFold over `groups` (true query identity)."""
    X = df[feats].to_numpy(np.float32); y = df[label].to_numpy()
    seeds = list(seeds) if seeds else [None]
    oof = np.zeros(len(df), dtype=np.float64); iters = []; models = []
    for sd in seeds:
        kw_s = dict(kw)
        if sd is not None and kind == "lgb":
            p = dict(kw_s.get("params") or {})
            p["seed"] = int(sd); p["bagging_seed"] = int(sd); p["feature_fraction_seed"] = int(sd)
            kw_s["params"] = p
        for tr, va in GroupKFold(n_splits).split(X, y, groups):
            m = make_model(kind, **kw_s).fit(X[tr], y[tr], groups[tr], feats)
            oof[va] += m.predict(X[va]) / len(seeds)
            iters.append(getattr(m, "best_iter", 0))
            if return_models:
                models.append(m)
    return (oof, iters, models) if return_models else (oof, iters)


def predict_bag(models, X):
    """Average probability predictions across a list of bagged fold/seed models."""
    out = np.zeros(len(X), dtype=np.float64)
    for m in models:
        out += m.predict(X) / len(models)
    return out


def fit_full(df, feats, label, groups, kind="lgb", **kw):
    X = df[feats].to_numpy(np.float32); y = df[label].to_numpy()
    return make_model(kind, **kw).fit(X, y, groups, feats)
