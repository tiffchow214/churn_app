# app.py
# E-commerce churn predictor (Gradio) — Hugging Face Space

from pathlib import Path
import json, joblib, numpy as np, pandas as pd, gradio as gr

# ---------- Paths (absolute, relative to this file) ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "churn_pipe.joblib"
META_PATH = HERE / "clean_meta_ready.json"
DRIVERS_JSON_PATH = HERE / "model_drivers.json"  

# ---------- Safe loaders ----------
def load_meta():
    """Load meta JSON or return safe defaults so the UI still boots."""
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            print(f">> Loaded meta from {META_PATH}")
            return meta
    except Exception as e:
        print("Meta load warning:", e)
        return {
            "target_col": "Churn",
            "numeric_cols": [],
            "categorical_cols": [],
            "ui_options": {},
            "feature_thresholds": {},
            "decision_threshold": 0.5,
        }

def load_model():
    try:
        if MODEL_PATH.exists():
            print(f">> Loading model from {MODEL_PATH}")
            return joblib.load(MODEL_PATH)
        print("Model not found:", MODEL_PATH)
    except Exception as e:
        print("Model load warning:", e)
    return None

def load_drivers_df() -> pd.DataFrame:
    """Load drivers from model_drivers.json.
    Supports:
      - {"table": [...]}
      - {"top_positive": [...], "top_negative": [...]}
      - [ {...}, {...} ]
    """
    try:
        if not DRIVERS_JSON_PATH.exists():
            print("Drivers file not found:", DRIVERS_JSON_PATH)
            return pd.DataFrame(columns=["feature", "coef", "odds_ratio"])

        with open(DRIVERS_JSON_PATH, "r") as f:
            payload = json.load(f)

        rows = []
        if isinstance(payload, dict):
            if "table" in payload and isinstance(payload["table"], list):
                rows = payload["table"]
            elif ("top_positive" in payload) or ("top_negative" in payload):
                rows = payload.get("top_positive", []) + payload.get("top_negative", [])
            else:
                # fall back: first list-of-dicts inside dict
                for v in payload.values():
                    if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                        rows = v
                        break
        elif isinstance(payload, list):
            rows = payload

        df = pd.DataFrame(rows)
        keep = [c for c in ["feature", "coef", "odds_ratio"] if c in df.columns]
        df = df[keep] if keep else df
        print(f">> Loaded drivers: {len(df)} rows from {DRIVERS_JSON_PATH}")
        return df
    except Exception as e:
        print("Drivers load warning:", e)
        return pd.DataFrame(columns=["feature", "coef", "odds_ratio"])

# ---------- Load artifacts once ----------
meta = load_meta()
pipe = load_model()
drivers_df = load_drivers_df()

thr_f2 = float(meta.get("decision_threshold", 0.5))   # recall-leaning threshold from training
thr_profit_default = 0.39                              # ROI threshold (you computed earlier)
feat_thr = meta.get("feature_thresholds", {})
ui = meta.get("ui_options", {})

# Dropdown choices (fallback if ui_options missing)
ORDER_CATS = ui.get(
    "PreferedOrderCat",
    ["Mobile Phone", "Fashion", "Laptop & Accessory", "Others", "Grocery"],
)
MARITALS = ui.get("MaritalStatus", ["Single", "Married", "Divorced"])

# Expected columns from training
NUM_COLS = meta.get("numeric_cols", [])
CAT_COLS = meta.get("categorical_cols", [])
EXPECTED_COLS = NUM_COLS + CAT_COLS
TARGET = meta.get("target_col", "Churn")

# ---------- Input schema for the UI ----------
REQUIRED_COLS = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "DaySinceLastOrder",
    "CashbackAmount",
    "PreferedOrderCat",
]

# ---------- Feature engineering to match training ----------
def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate *_was_missing flags as during cleaning."""
    out = df.copy()
    for base in ["DaySinceLastOrder", "Tenure", "WarehouseToHome"]:
        flag = f"{base}_was_missing"
        if flag not in out.columns:
            out[flag] = out[base].isna().astype(int) if base in out.columns else 1
    return out

def normalize_complain(col_like) -> pd.Series:
    """Accept Yes/No/True/False/1/0 → 0/1."""
    def _to01(v):
        s = str(v).strip().lower()
        if s in {"1", "yes", "y", "true", "t"}:
            return 1
        return 0
    return pd.Series([_to01(v) for v in col_like], index=getattr(col_like, "index", None))

def engineer(df_row: pd.DataFrame) -> pd.DataFrame:
    """Engineer the same features you used in training."""
    r = df_row.copy()
    t_early = int(feat_thr.get("tenure_early_max", 6))
    d_far = float(feat_thr.get("distance_far_min", 15))
    sat_low = int(feat_thr.get("sat_low_max", 3))
    cb_q20 = float(feat_thr.get("cashback_q20", r["CashbackAmount"].quantile(0.20)))

    # EarlyTenure & TenureBand
    if "Tenure" in r:
        r["Tenure"] = pd.to_numeric(r["Tenure"], errors="coerce")
        r["EarlyTenure"] = (r["Tenure"] <= t_early).astype(int)
        r["TenureBand"] = pd.cut(
            r["Tenure"], bins=[-0.1, 6, 12, 24, np.inf],
            labels=["≤6", "6–12", "12–24", ">24"]
        ).astype(str)

    # Distance friction
    if "WarehouseToHome" in r:
        r["WarehouseToHome"] = pd.to_numeric(r["WarehouseToHome"], errors="coerce")
        r["FarFromWarehouse"] = (r["WarehouseToHome"] >= d_far).astype(int)

    # Low cashback
    if "CashbackAmount" in r:
        r["CashbackAmount"] = pd.to_numeric(r["CashbackAmount"], errors="coerce")
        r["LowCashback"] = (r["CashbackAmount"] <= cb_q20).astype(int)

    # Devices per address (avoid 0)
    if "NumberOfDeviceRegistered" in r and "NumberOfAddress" in r:
        num = pd.to_numeric(r["NumberOfDeviceRegistered"], errors="coerce")
        den = pd.to_numeric(r["NumberOfAddress"], errors="coerce").clip(lower=1)
        r["DevicesPerAddress"] = num / den

    # Low satisfaction & interaction
    if "SatisfactionScore" in r:
        r["SatisfactionScore"] = pd.to_numeric(r["SatisfactionScore"], errors="coerce")
        r["LowSatisfaction"] = (r["SatisfactionScore"] <= sat_low).astype(int)

    if "Complain" in r:
        comp = normalize_complain(r["Complain"])
        r["Complain"] = comp
        if "LowSatisfaction" in r:
            r["Complain_LowSat"] = ((comp == 1) & (r["LowSatisfaction"] == 1)).astype(int)

    # Normalize category text (Mobile → Mobile Phone)
    if "PreferedOrderCat" in r:
        r["PreferedOrderCat"] = (
            r["PreferedOrderCat"]
            .astype(str).str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"(?i)^mobile$", "Mobile Phone", regex=True)
        )

    return r

def align_to_training_columns(df_any: pd.DataFrame) -> pd.DataFrame:
    """Make sure all expected training-time columns exist and types are sane."""
    out = df_any.copy()
    for c in EXPECTED_COLS:
        if c not in out.columns:
            out[c] = 0 if c in NUM_COLS else ""
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    for c in CAT_COLS:
        out[c] = out[c].astype(str).fillna("")
    return out[EXPECTED_COLS]

def flags_for_row(s: pd.Series) -> str:
    msgs = []
    if s.get("EarlyTenure", 0) == 1:       msgs.append("Early tenure (≤6 mo)")
    if s.get("Complain", 0) == 1:          msgs.append("Recent complaint")
    if s.get("LowCashback", 0) == 1:       msgs.append("Low cashback (≤Q20)")
    if s.get("FarFromWarehouse", 0) == 1:  msgs.append("Far from warehouse (≥15 km)")
    if s.get("Complain_LowSat", 0) == 1:   msgs.append("Complaint + Low satisfaction")
    return " | ".join(msgs) if msgs else "No risk flags"

def choose_threshold(mode: str, custom_thr: float, profit_thr: float) -> float:
    if mode == "Recall (F2)":
        return float(thr_f2)
    if mode == "Profit (ROI)":
        return float(profit_thr)
    return float(custom_thr)

# ---------- Predictors ----------
def predict_fn(
    Tenure, WarehouseToHome, NumberOfDeviceRegistered, SatisfactionScore,
    MaritalStatus, NumberOfAddress, Complain, DaySinceLastOrder,
    CashbackAmount, PreferedOrderCat, threshold_mode, custom_threshold, profit_threshold
):
    if pipe is None:
        return "Model file missing", "N/A", "Ask the owner to upload churn_pipe.joblib"

    row = pd.DataFrame([{
        "Tenure": Tenure,
        "WarehouseToHome": WarehouseToHome,
        "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
        "SatisfactionScore": SatisfactionScore,
        "MaritalStatus": MaritalStatus,
        "NumberOfAddress": NumberOfAddress,
        "Complain": Complain,  # "Yes"/"No" handled in engineer()
        "DaySinceLastOrder": DaySinceLastOrder,
        "CashbackAmount": CashbackAmount,
        "PreferedOrderCat": PreferedOrderCat,
    }])

    row = add_missing_flags(row)
    row = engineer(row)
    X = align_to_training_columns(row)

    prob = float(pipe.predict_proba(X)[:, 1][0])
    thr = choose_threshold(threshold_mode, custom_threshold, profit_threshold)
    decision = "Flag (likely churn)" if prob >= thr else "Do not flag"
    why = flags_for_row(row.iloc[0])

    return f"{prob:.3f}", decision, why

def batch_predict_fn(file, threshold_mode, custom_threshold, profit_threshold):
    if pipe is None:
        return pd.DataFrame(), None

    try:
        df_in = pd.read_csv(file.name)
    except Exception as e:
        return pd.DataFrame({"error": [f"Failed to read CSV: {e}"]}), None

    if "Complain" in df_in.columns:
        df_in["Complain"] = normalize_complain(df_in["Complain"])
    else:
        df_in["Complain"] = 0

    df_in = add_missing_flags(df_in)
    df_in = engineer(df_in)
    X = align_to_training_columns(df_in)

    probs = pipe.predict_proba(X)[:, 1]
    thr = choose_threshold(threshold_mode, custom_threshold, profit_threshold)
    decision = np.where(probs >= thr, "Flag (likely churn)", "Do not flag")

    whys = [flags_for_row(s) for _, s in df_in.iterrows()]

    out = df_in.copy()
    out["churn_prob"] = probs.round(4)
    out["decision"] = decision
    out["why"] = whys

    out_path = Path("scored.csv")
    out.to_csv(out_path, index=False)

    preview = out.head(25)
    return preview, str(out_path)

# ---------- Gradio UI ----------
with gr.Blocks(title="E-commerce Churn Predictor") as demo:
    gr.Markdown("## E-commerce Churn Predictor\nGet churn probability, a decision flag, and **why** (risk flags).")

    with gr.Tab("Single prediction"):
        with gr.Row():
            with gr.Column():
                Tenure = gr.Slider(0, 60, value=6, step=1, label="Tenure (months)")
                WarehouseToHome = gr.Slider(0, 50, value=10, step=1, label="Warehouse → Home (km)")
                NumberOfDeviceRegistered = gr.Slider(1, 6, value=3, step=1, label="# Devices registered")
                SatisfactionScore = gr.Slider(1, 5, value=3, step=1, label="Satisfaction score (1–5)")
                NumberOfAddress = gr.Slider(1, 10, value=2, step=1, label="# Addresses on file")
                DaySinceLastOrder = gr.Slider(0, 90, value=7, step=1, label="Days since last order")
                CashbackAmount = gr.Slider(0, 350, value=150, step=1, label="Cashback amount")
                Complain = gr.Radio(choices=["No", "Yes"], value="No", label="Complained?")
                PreferedOrderCat = gr.Dropdown(ORDER_CATS, value=ORDER_CATS[0], label="Preferred order category")
                MaritalStatus = gr.Dropdown(MARITALS, value=MARITALS[0], label="Marital status")

                gr.Markdown("**Decision threshold**")
                threshold_mode   = gr.Radio(["Recall (F2)", "Profit (ROI)", "Custom"], value="Recall (F2)")
                profit_threshold = gr.Slider(0.05, 0.95, step=0.01, value=thr_profit_default, label="Profit threshold")
                custom_threshold = gr.Slider(0.05, 0.95, step=0.01, value=thr_f2, label="Custom threshold")

                btn_single = gr.Button("Predict")

            with gr.Column():
                prob = gr.Textbox(label="Churn probability", interactive=False)
                dec  = gr.Textbox(label="Decision", interactive=False)
                expl = gr.Textbox(label="Why this decision (risk flags)", lines=6, interactive=False)

        btn_single.click(
            predict_fn,
            inputs=[Tenure, WarehouseToHome, NumberOfDeviceRegistered, SatisfactionScore,
                    MaritalStatus, NumberOfAddress, Complain, DaySinceLastOrder,
                    CashbackAmount, PreferedOrderCat, threshold_mode, custom_threshold, profit_threshold],
            outputs=[prob, dec, expl]
        )

    with gr.Tab("Batch (CSV)"):
        gr.Markdown(
            "Upload a CSV with these columns:\n\n"
            f"`{', '.join(REQUIRED_COLS)}`\n\n"
            "Tip: `Complain` can be **Yes/No** or **1/0**."
        )
        file_in = gr.File(file_types=[".csv"], label="Upload CSV")
        thr_mode_b = gr.Radio(["Recall (F2)", "Profit (ROI)", "Custom"], value="Recall (F2)", label="Decision threshold")
        profit_thr_b = gr.Slider(0.05, 0.95, step=0.01, value=thr_profit_default, label="Profit threshold")
        custom_thr_b = gr.Slider(0.05, 0.95, step=0.01, value=thr_f2, label="Custom threshold")
        btn_batch = gr.Button("Score file")

        preview_out = gr.Dataframe(label="Preview (first 25 rows)", interactive=False)
        file_out = gr.File(label="Download scored CSV")

        btn_batch.click(
            batch_predict_fn,
            inputs=[file_in, thr_mode_b, custom_thr_b, profit_thr_b],
            outputs=[preview_out, file_out]
        )

    with gr.Tab("Model drivers"):
        gr.Markdown("Top global drivers from a transparent logistic refit (odds ratios shown).")
        gr.Dataframe(value=drivers_df, interactive=False, wrap=True, label="Global drivers")

# On Spaces, a plain launch() is perfect
if __name__ == "__main__":
    demo.launch()
