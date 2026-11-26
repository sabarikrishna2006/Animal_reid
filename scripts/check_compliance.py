"""Static scan: only allowlisted imports, only allowlisted cv2 attributes (matching + geometric verification)."""
import ast
import pathlib
import sys

ALLOWED_THIRD_PARTY = {"numpy", "pandas", "scipy", "sklearn", "statsmodels", "imblearn", "xgboost",
                       "lightgbm", "catboost", "networkx", "cv2", "faiss", "matplotlib", "seaborn"}
LOCAL = {"reid", "src", "scripts", "tests", "check_compliance",
         "core", "lnbnn", "features", "model2", "exp_kpg", "exp_kpg_v2",
         "exp_lnb_img", "exp_blend_check", "make_kpg_probs", "make_dev_bgclean",
         "build_dev_v10", "make_dev_bgclean2", "make_dev_bgclean3", "make_dev_hubness",
         "make_dev_v10_fullbg", "make_dev_v10_lprop", "make_dev_v10_lda512", "exp_dev_learned_prop"}
CV2_ALLOWED = {"BFMatcher", "BFMatcher_create", "FlannBasedMatcher", "DMatch", "NORM_L2",
               "findHomography", "estimateAffinePartial2D", "estimateAffine2D", "findFundamentalMat",
               "perspectiveTransform", "transform", "RANSAC", "USAC_MAGSAC", "USAC_DEFAULT", "LMEDS",
               "setNumThreads", "setRNGSeed", "__version__"}
STDLIB = set(sys.stdlib_module_names)
FORBIDDEN_STDLIB_LIKE = {"joblib"}  # not stdlib anyway; listed for clarity


def scan(root="."):
    errors = []
    for path in pathlib.Path(root).rglob("*.py"):
        if any(p in path.parts for p in (".venv", "venv", "site-packages", "eda")):
            continue  # eda/ = exploratory scripts (also compliant, but not part of the pipeline)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m not in ALLOWED_THIRD_PARTY | STDLIB | LOCAL or m in FORBIDDEN_STDLIB_LIKE:
                    errors.append(f"{path}:{node.lineno} forbidden import '{m}'")
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "cv2" and node.attr not in CV2_ALLOWED):
                errors.append(f"{path}:{node.lineno} cv2.{node.attr} not allowed")
            if isinstance(node, ast.ImportFrom) and node.module == "cv2":
                errors.append(f"{path}:{node.lineno} use 'import cv2' + allowlisted attributes only")
    return errors


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    errors = scan(root)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("compliance OK")


if __name__ == "__main__":
    main()
