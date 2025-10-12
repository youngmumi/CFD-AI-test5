# airfoil_pipeline.py
# AoA + CST 계수 -> Cl, Cd 예측
# 1) 데이터 분석 그래프  2) 모델 학습/저장  3) 성능 지표/잔차/중요도 그래프
# matplotlib만 사용(의존성 최소), 실행 옵션은 --help 참고

import argparse
import json
import os
import re
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======= 사용자 지정 기본값 (여기만 바꾸면 됩니다) =======
CSV_PATH = r"C:\Users\yujeong.DESKTOP-LNFC8O3\Documents\coding work\CFD-AI-5\aiaa_airfoil_training.csv"
OUTDIR = "outputs"
N_ESTIMATORS = 200
MAX_DEPTH = 20
SEED = 42
# ======================================================

os.makedirs(OUTDIR, exist_ok=True)

# --------------------------
# 유틸
# --------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """특성: AoA, CST*  / 타깃: Cl, Cd 자동 탐지"""
    cols = df.columns.tolist()

    # Targets
    tgt = []
    for c in cols:
        if re.fullmatch(r"(?i)cl|c_l", c):  # Cl
            tgt.append(c)
        if re.fullmatch(r"(?i)cd|c_d", c):  # Cd
            tgt.append(c)
    # contains fallback
    if len(tgt) < 2:
        for c in cols:
            if re.search(r"(?i)\bcl\b", c) and c not in tgt:
                tgt.append(c)
            if re.search(r"(?i)\bcd\b", c) and c not in tgt:
                tgt.append(c)
    # 정렬: [Cl, Cd]
    def tkey(x):
        if re.search(r"(?i)cl", x): return 0
        if re.search(r"(?i)cd", x): return 1
        return 2
    tgt = sorted(list(dict.fromkeys(tgt)), key=tkey)[:2]

    # Features
    feats = []
    for c in cols:
        if c in tgt:
            continue
        if re.fullmatch(r"(?i)aoa", c):
            feats.append(c)
        if re.fullmatch(r"(?i)cst\d+", c):
            feats.append(c)
    # fallback: 모든 수치형 중 타깃 제외
    if not feats:
        numerics = df.select_dtypes(include=[np.number]).columns.tolist()
        feats = [c for c in numerics if c not in tgt]

    # 중복 제거(순서 유지)
    feats = list(dict.fromkeys(feats))
    return feats, tgt

def train_test_split_idx(n: int, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_ratio))
    return idx[n_test:], idx[:n_test]  # train_idx, test_idx

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

# --------------------------
# 1) 데이터 분석 그래프
# --------------------------
def plot_data_analysis(df: pd.DataFrame, features: List[str], targets: List[str], outdir: str):
    ensure_dir(outdir)

    # 1-1. AoA vs Cl, AoA vs Cd 산점도
    aoa_cols = [c for c in features if re.fullmatch(r"(?i)aoa", c)]
    if aoa_cols:
        aoa = aoa_cols[0]
        for t in targets:
            plt.figure()
            plt.scatter(df[aoa], df[t], s=8)
            plt.xlabel(aoa); plt.ylabel(t)
            plt.title(f"{aoa} vs {t}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"scatter_{aoa}_vs_{t}.png"), dpi=150)
            plt.close()

    # 1-2. 타깃 분포 히스토그램
    plt.figure()
    for t in targets:
        plt.hist(df[t].values, bins=40, alpha=0.6, label=t)
    plt.title("Distribution of targets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "target_distribution.png"), dpi=150)
    plt.close()

    # 1-3. 상관행렬(수치형 전체)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True).values
        labels = num_df.select_dtypes(include=[np.number]).columns.tolist()

        plt.figure(figsize=(max(6, len(labels)*0.4), max(5, len(labels)*0.4)))
        im = plt.imshow(corr, interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=150)
        plt.close()

# --------------------------
# 2) 모델 학습/저장 (RandomForestRegressor)
# --------------------------
def train_model(df: pd.DataFrame, features: List[str], targets: List[str], outdir: str,
                n_estimators: int = 300, max_depth: int = 20, seed: int = 42):
    from sklearn.ensemble import RandomForestRegressor

    X = df[features].to_numpy(dtype=float)
    y = df[targets].to_numpy(dtype=float)

    tr_idx, te_idx = train_test_split_idx(len(df), test_ratio=0.2, seed=seed)
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 지표
    metrics = {
        "RMSE_Cl": rmse(y_test[:, 0], y_pred[:, 0]),
        "R2_Cl": r2(y_test[:, 0], y_pred[:, 0]),
        "RMSE_Cd": rmse(y_test[:, 1], y_pred[:, 1]),
        "R2_Cd": r2(y_test[:, 1], y_pred[:, 1]),
    }

    # 아티팩트 저장
    ensure_dir(outdir)
    model_path = os.path.join(outdir, "airfoil_rf_model.pkl")
    bundle = {"model": model, "features": features, "targets": targets}
    joblib.dump(bundle, model_path)

    # 예측 CSV
    pred_df = pd.DataFrame(
        np.hstack([y_test, y_pred]),
        columns=[f"true_{t}" for t in targets] + [f"pred_{t}" for t in targets]
    )
    pred_csv = os.path.join(outdir, "test_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    # 피처 중요도
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        plt.figure(figsize=(max(6, len(features)*0.35), 4.5))
        imp.plot(kind="bar")
        plt.title("Feature Importance (RandomForest)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150)
        plt.close()

    # y_true vs y_pred
    for i, t in enumerate(targets):
        yt, yp = y_test[:, i], y_pred[:, i]
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        plt.figure()
        plt.scatter(yt, yp, s=10)
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(f"True {t}"); plt.ylabel(f"Predicted {t}")
        plt.title(f"True vs Predicted ({t})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"true_vs_pred_{t}.png"), dpi=150)
        plt.close()

    # Residuals
    res = y_test - y_pred
    for i, t in enumerate(targets):
        plt.figure()
        plt.scatter(y_pred[:, i], res[:, i], s=10)
        plt.axhline(0)
        plt.xlabel(f"Predicted {t}"); plt.ylabel("Residual")
        plt.title(f"Residuals vs Predicted ({t})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"residuals_{t}.png"), dpi=150)
        plt.close()

    # 메트릭 저장
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "model_path": model_path,
        "pred_csv": pred_csv,
        "metrics": metrics
    }

# --------------------------
# 3) 보고서 모드: 저장된 예측 CSV로 성능 그래프 재출력
# --------------------------
def report_from_predictions(pred_csv: str, outdir: str, targets: List[str] = None):
    ensure_dir(outdir)
    df = pd.read_csv(pred_csv)
    if targets is None:
        tgs = []
        for c in df.columns:
            m = re.match(r"true_(.+)", c)
            if m:
                tgs.append(m.group(1))
        targets = list(dict.fromkeys(tgs)) or ["Cl", "Cd"]

    for t in targets:
        yt = df[f"true_{t}"].values
        yp = df[f"pred_{t}"].values
        _rmse = rmse(yt, yp)
        _r2 = r2(yt, yp)

        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        plt.figure()
        plt.scatter(yt, yp, s=10)
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(f"True {t}"); plt.ylabel(f"Predicted {t}")
        plt.title(f"True vs Predicted ({t}) — RMSE={_rmse:.4g}, R²={_r2:.4g}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"report_true_vs_pred_{t}.png"), dpi=150)
        plt.close()

        res = yt - yp
        plt.figure()
        plt.scatter(yp, res, s=10)
        plt.axhline(0)
        plt.xlabel(f"Predicted {t}"); plt.ylabel("Residual")
        plt.title(f"Residuals vs Predicted ({t})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"report_residuals_{t}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.hist(res, bins=40)
        plt.title(f"Residual Histogram ({t})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"report_residual_hist_{t}.png"), dpi=150)
        plt.close()

# --------------------------
# 메인
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="AoA + CST -> Cl, Cd surrogate pipeline")
    # 기본값을 상단 상수로 연결
    ap.add_argument("--csv", type=str, default=CSV_PATH, help="Input CSV path (default: CSV_PATH)")
    ap.add_argument("--outdir", type=str, default=OUTDIR, help="Output directory")
    ap.add_argument("--n_estimators", type=int, default=N_ESTIMATORS, help="RF trees")
    ap.add_argument("--max_depth", type=int, default=MAX_DEPTH, help="RF max depth")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap.add_argument("--report", action="store_true", help="Report mode (skip training)")
    ap.add_argument("--pred_csv", type=str, help="Predictions CSV for report mode")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # report 모드 처리
    if args.report:
        if not args.pred_csv:
            raise SystemExit("--report 모드에서는 --pred_csv 가 필요합니다.")
        report_from_predictions(args.pred_csv, args.outdir)
        print(f"[OK] Report images saved to: {args.outdir}")
        return

    # --csv 생략 시에도 기본 CSV_PATH 사용
    csv_path = args.csv or CSV_PATH
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # 데이터 로드 & 전처리
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.strip().replace(" ", "_") for c in df.columns}, inplace=True)

    features, targets = detect_columns(df)
    if len(targets) < 2:
        raise SystemExit("타깃(Cl, Cd) 컬럼을 찾을 수 없습니다. CSV 헤더를 확인하세요.")

    use_cols = list(dict.fromkeys(features + targets))
    df = df[use_cols].dropna()

    # 1) 데이터 분석 그래프
    plot_data_analysis(df, features, targets, args.outdir)

    # 2) 학습/저장/지표 및 성능 그래프
    results = train_model(
        df, features, targets, args.outdir,
        n_estimators=args.n_estimators, max_depth=args.max_depth, seed=args.seed
    )

    print(json.dumps({
        "features": features,
        "targets": targets,
        "model_path": results["model_path"],
        "pred_csv": results["pred_csv"],
        "metrics": results["metrics"],
        "outdir": args.outdir
    }, ensure_ascii=False, indent=2))

    print(f"[OK] 모든 결과물이 '{args.outdir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
