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
            p["monotone_constraints"] = [1 if f in MONO_POS else 0 for f in feats]
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
        self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                   MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                 validation_fraction=0.15, max_iter=200, random_state=2026))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def make_model(kind, **kw):
    return {"lgb": LGBModel, "lr": LRModel, "xgb": XGBModel, "cat": CatModel, "mlp": MLPModel}[kind](**kw)


def oof_predict(df, feats, label, groups, kind="lgb", n_splits=5, **kw):
    """Out-of-fold predictions with GroupKFold over `groups` (true query identity)."""
    X = df[feats].to_numpy(np.float32); y = df[label].to_numpy()
    oof = np.zeros(len(df)); iters = []
    for tr, va in GroupKFold(n_splits).split(X, y, groups):
        m = make_model(kind, **kw).fit(X[tr], y[tr], groups[tr], feats)
        oof[va] = m.predict(X[va]); iters.append(getattr(m, "best_iter", 0))
    return oof, iters


def fit_full(df, feats, label, groups, kind="lgb", **kw):
    X = df[feats].to_numpy(np.float32); y = df[label].to_numpy()
    return make_model(kind, **kw).fit(X, y, groups, feats)
