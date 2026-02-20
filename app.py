from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

# -----------------------------
# Helpers
# -----------------------------
def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def to_seconds_total_time(col: pd.Series) -> np.ndarray:
    """
    Converts "Total Time" column (HH:MM:SS or timedeltas) to seconds.
    """
    td = pd.to_timedelta(col.astype(str), errors="coerce")
    out = td.dt.total_seconds().to_numpy()
    return out


def load_record_any_excel(file_like) -> pd.DataFrame:
    """
    Loads 'record' sheet if exists, else first sheet.
    Standardizes to columns:
    t_s, dt_s, current_a, voltage_v, step_type, soc_pct, capacity_ah, dchg_cap_ah
    """
    xls = pd.ExcelFile(file_like)
    sheet = "record" if "record" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(file_like, sheet_name=sheet)

    # Column mapping (common in your files)
    colmap = {
        "Total Time": "total_time",
        "Current(A)": "current_a",
        "Voltage(V)": "voltage_v",
        "Step Type": "step_type",
        "SOC/DOD(%)": "soc_pct",
        "Capacity(Ah)": "capacity_ah",
        "DChg. Cap.(Ah)": "dchg_cap_ah",
        "Chg. Cap.(Ah)": "chg_cap_ah",
        "Time": "time",
        "Date": "date",
    }

    # Create standardized fields if present
    out = pd.DataFrame()
    for k, v in colmap.items():
        if k in df.columns:
            out[v] = df[k]

    # time
    if "total_time" in out.columns:
        t_s = to_seconds_total_time(out["total_time"])
    elif "time" in out.columns:
        t_s = to_seconds_total_time(out["time"])
    else:
        # fallback index-based
        t_s = np.arange(len(df), dtype=float)

    out["t_s"] = t_s

    # required: current, voltage
    if "current_a" not in out.columns or "voltage_v" not in out.columns:
        raise ValueError(
            "This file doesn't look like a battery record export. "
            "Expected columns like Current(A) and Voltage(V) in the record sheet."
        )

    out["current_a"] = safe_num(out["current_a"])
    out["voltage_v"] = safe_num(out["voltage_v"])

    # step type
    if "step_type" in out.columns:
        out["step_type"] = out["step_type"].astype(str)
    else:
        out["step_type"] = "unknown"

    # SOC
    if "soc_pct" in out.columns:
        out["soc_pct"] = safe_num(out["soc_pct"])
    else:
        out["soc_pct"] = np.nan

    # Capacity
    if "capacity_ah" in out.columns:
        out["capacity_ah"] = safe_num(out["capacity_ah"])
    else:
        out["capacity_ah"] = np.nan

    if "dchg_cap_ah" in out.columns:
        out["dchg_cap_ah"] = safe_num(out["dchg_cap_ah"])
    else:
        out["dchg_cap_ah"] = np.nan

    # clean
    out = out.dropna(subset=["t_s", "current_a", "voltage_v"]).copy()
    out = out.sort_values("t_s").reset_index(drop=True)
    out = out[~out["t_s"].duplicated(keep="first")].reset_index(drop=True)

    dt = np.diff(out["t_s"].to_numpy(), prepend=out["t_s"].iloc[0])
    dt[0] = np.nan
    out["dt_s"] = dt

    return out


def make_interp(x, y, kind="linear"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        raise ValueError("Not enough points to interpolate.")
    idx = np.argsort(x)
    return interp1d(x[idx], y[idx], kind=kind, fill_value="extrapolate", bounds_error=False)


def append_coulomb_counting_soc(df: pd.DataFrame, Q_AH: float, soc0: float = None) -> pd.DataFrame:
    """
    Calculates true SOC using precise Ah integration (Coulomb Counting).
    If soc0 is None, it guesses based on the initial voltage.
    """
    df = df.copy()
    I = df["current_a"].to_numpy(float)
    dt = df["dt_s"].to_numpy(float)
    dt = np.nan_to_num(dt, nan=0.0)
    
    # Auto-guess initial SOC based on first voltage if not provided
    if soc0 is None:
        first_v = df["voltage_v"].iloc[0]
        soc0 = 1.0 if first_v > 4.0 else 0.0 # Heuristic for NMC
        
    # Ah integrated over time: dt is in seconds, so divide by 3600
    # For robust integration we need to ensure discharge and charge are correct
    # Usually Neware/Arbin have Charge=+ and Discharge=- in Current(A). 
    ah_integrated = np.cumsum(I * dt) / 3600.0
    
    soc_cc = soc0 + (ah_integrated / Q_AH)
    df["soc_cc"] = np.clip(soc_cc, 0.0, 1.0)
    
    return df


def extract_rest_ocv_points(df: pd.DataFrame, Q_AH: float, rest_thresh_a: float = 0.02, min_rest_s: float = 120.0, tail_take_s: float = 30.0):
    """
    Extract rest tail points (SOC, Voltage) using Coulomb counted SOC.
    Returns arrays of (soc01, v).
    """
    # Ensure we have a reliable SOC column computed
    df = append_coulomb_counting_soc(df, Q_AH)
    
    t = df["t_s"].to_numpy(float)
    I = df["current_a"].to_numpy(float)
    V = df["voltage_v"].to_numpy(float)
    soc = df["soc_cc"].to_numpy(float)

    step = df["step_type"].astype(str).str.lower().to_numpy()
    if np.any(step != "unknown"):
        rest_mask = step == "rest"
    else:
        rest_mask = np.abs(I) <= rest_thresh_a

    # find segments of rest_mask
    segs = []
    start = None
    for i, m in enumerate(rest_mask):
        if m and start is None:
            start = i
        if (not m or i == len(rest_mask) - 1) and start is not None:
            end = i if not m else i + 1
            segs.append((start, end))
            start = None

    soc_pts = []
    v_pts = []

    for s, e in segs:
        if e - s < 2:
            continue
        dur = t[e - 1] - t[s]
        if not np.isfinite(dur) or dur < min_rest_s:
            continue

        # tail slice: last tail_take_s seconds
        t_end = t[e - 1]
        take_mask = (t[s:e] >= (t_end - tail_take_s))
        if take_mask.sum() < 2:
            continue

        v_tail = V[s:e][take_mask]
        soc_tail = soc[s:e][take_mask]

        # use medians to reduce noise
        v_med = float(np.nanmedian(v_tail))
        soc_med = float(np.nanmedian(soc_tail))

        if np.isfinite(v_med) and np.isfinite(soc_med):
            soc_pts.append(clamp01(soc_med))
            v_pts.append(v_med)

    return np.array(soc_pts, float), np.array(v_pts, float)


def enforce_monotonic_inverse(tab: pd.DataFrame):
    """
    Builds inverse SOC(OCV) interpolation safely.
    OCV is not always strictly monotonic if data is noisy; we enforce monotonic by sorting by voltage
    and removing duplicates.
    """
    soc = tab["soc"].to_numpy(float)
    v = tab["ocv_v"].to_numpy(float)

    # sort by voltage
    idx = np.argsort(v)
    v = v[idx]
    soc = soc[idx]

    # drop duplicate voltages
    keep = np.r_[True, np.diff(v) > 1e-6]
    v = v[keep]
    soc = soc[keep]

    soc = np.clip(soc, 0, 1)

    inv = interp1d(v, soc, kind="linear", fill_value="extrapolate", bounds_error=False)
    return inv


def estimate_capacity_ah(df_any: pd.DataFrame) -> float | None:
    """
    Estimate Q_AH from available columns.
    Prefer DChg. Cap.(Ah) max, else Capacity(Ah) max.
    """
    if "dchg_cap_ah" in df_any.columns and np.isfinite(df_any["dchg_cap_ah"]).any():
        q = float(np.nanmax(df_any["dchg_cap_ah"].to_numpy(float)))
        if np.isfinite(q) and q > 0:
            return q
    if "capacity_ah" in df_any.columns and np.isfinite(df_any["capacity_ah"]).any():
        q = float(np.nanmax(df_any["capacity_ah"].to_numpy(float)))
        if np.isfinite(q) and q > 0:
            return q
    return None


def detect_discharge_sign(df: pd.DataFrame) -> int:
    """
    Returns sign_flip: multiply current by sign_flip so that discharge is positive internally.
    """
    step = df["step_type"].astype(str).str.lower()
    I = df["current_a"].to_numpy(float)

    mask_dchg = step == "cc dchg"
    if mask_dchg.any():
        med = float(np.nanmedian(I[mask_dchg.to_numpy()]))
    else:
        # fallback: use larger magnitude median
        med = float(np.nanmedian(I))

    # if discharge current is negative, flip sign
    return -1 if med < 0 else 1


# -----------------------------
# 2RC + H EKF
# State: x = [soc, v1, v2, h]
# Measurement: V = OCV(soc) + h - I*R0 - v1 - v2
# Dynamics:
#   soc_{k+1} = soc_k - I*dt/(Q*3600)
#   v1_{k+1} = a1 v1_k + (1-a1) R1 I
#   v2_{k+1} = a2 v2_k + (1-a2) R2 I
#   h_{k+1}  = ah h_k + (1-ah) k_h * sign(I)   (simple hysteresis)
# -----------------------------
def run_ekf_2rc_h(
    df: pd.DataFrame,
    ocv_fn,
    docv_fn,
    R0: float, R1: float, C1: float, R2: float, C2: float,
    Q_AH: float,
    soc0: float,
    k_h: float,
    tau_h: float,
    q_soc: float,
    q_v: float,
    q_h: float,
    r_v: float,
    sign_flip: int,
):
    t = df["t_s"].to_numpy(float)
    dt = df["dt_s"].to_numpy(float)
    I_raw = df["current_a"].to_numpy(float)
    V_meas = df["voltage_v"].to_numpy(float)

    # discharge positive internal
    I = sign_flip * I_raw

    n = len(t)
    x = np.zeros((n, 4), float)  # soc,v1,v2,h
    P = np.diag([0.05, 0.02, 0.02, 0.02])**2

    # initial
    x[0, 0] = clamp01(soc0)
    x[0, 1] = 0.0
    x[0, 2] = 0.0
    x[0, 3] = 0.0

    Qk = np.diag([q_soc, q_v, q_v, q_h])  # process noise
    Rk = np.array([[r_v]], float)        # measurement noise

    V_pred = np.zeros(n, float)

    for k in range(1, n):
        if not np.isfinite(dt[k]) or dt[k] <= 0:
            x[k] = x[k-1]
            V_pred[k] = V_pred[k-1]
            continue

        soc, v1, v2, h = x[k-1]
        Ik = float(I[k])
        dtk = float(dt[k])

        # discrete params
        a1 = float(np.exp(-dtk / (R1 * C1)))
        a2 = float(np.exp(-dtk / (R2 * C2)))
        ah = float(np.exp(-dtk / max(tau_h, 1e-6)))
        sgn = 0.0
        if abs(Ik) > 1e-6:
            sgn = 1.0 if Ik > 0 else -1.0

        # ---- Predict ----
        soc_pred = clamp01(soc - (Ik * dtk) / (Q_AH * 3600.0))
        v1_pred = a1 * v1 + (1.0 - a1) * (R1 * Ik)
        v2_pred = a2 * v2 + (1.0 - a2) * (R2 * Ik)
        h_pred  = ah * h + (1.0 - ah) * (k_h * sgn)

        x_pred = np.array([soc_pred, v1_pred, v2_pred, h_pred], float)

        # Jacobian F = d f / d x
        F = np.eye(4, dtype=float)
        F[1, 1] = a1
        F[2, 2] = a2
        F[3, 3] = ah

        P_pred = F @ P @ F.T + Qk

        # ---- Measurement update ----
        ocv = float(ocv_fn(soc_pred))
        V_hat = ocv + h_pred - Ik * R0 - v1_pred - v2_pred
        V_pred[k] = V_hat

        # H = d h(x) / d x = [dOCV/dSOC, -1, -1, +1]
        dOCV = float(docv_fn(soc_pred))
        H = np.array([[dOCV, -1.0, -1.0, 1.0]], float)  # shape (1,4)

        # innovation covariance S = H P H^T + R
        S = (H @ P_pred @ H.T) + Rk  # shape (1,1)
        S_scalar = float(S[0, 0])

        if not np.isfinite(S_scalar) or S_scalar <= 1e-12:
            x[k] = x_pred
            P = P_pred
            continue

        # Kalman gain K = P H^T / S
        K = (P_pred @ H.T) / S_scalar  # shape (4,1)

        y = float(V_meas[k] - V_hat)   # innovation
        x_upd = x_pred + (K[:, 0] * y)

        # clamp SOC
        x_upd[0] = clamp01(float(x_upd[0]))

        # covariance update
        P = (np.eye(4) - K @ H) @ P_pred
        
        # --- CRITICAL FIX: Force symmetry to prevent EKF divergence ---
        P = (P + P.T) / 2.0  

        x[k] = x_upd

    soc_est = x[:, 0]
    v1_est = x[:, 1]
    v2_est = x[:, 2]
    h_est  = x[:, 3]

    rmse_v = float(np.sqrt(np.nanmean((V_meas - V_pred) ** 2)))

    return {
        "t_s": t,
        "I_raw": I_raw,
        "I_int": I,
        "V_meas": V_meas,
        "V_pred": V_pred,
        "soc_est": soc_est,
        "v1": v1_est,
        "v2": v2_est,
        "h": h_est,
        "rmse_v": rmse_v,
        "sign_flip": sign_flip,
    }


def build_soc_reference(df: pd.DataFrame, ref_mode: str, Q_AH: float, ocv_inv_fn=None):
    """
    Returns (t_ref, soc_ref, label) or (None,None,label) if unavailable.
    """
    t = df["t_s"].to_numpy(float)

    if ref_mode == "SOC/DOD(%) column":
        if df["soc_pct"].notna().any():
            soc = (df["soc_pct"].to_numpy(float) / 100.0)
            return t, np.clip(soc, 0, 1), "SOC/DOD(%) column (may be step label)"
        return None, None, "SOC/DOD(%) not available"

    if ref_mode == "Capacity-based":
        # Calculate it cleanly from scratch rather than relying on hardware columns
        df_cc = append_coulomb_counting_soc(df, Q_AH, soc0=1.0) # Assuming full charge at start. Adjust if needed.
        soc = df_cc["soc_cc"].to_numpy(float)
        return t, soc, f"Ah-counted SOC (Q={Q_AH:.4f}Ah)"

    if ref_mode == "OCV-rest inversion (tails)":
        if ocv_inv_fn is None:
            return None, None, "Need OCV inverse function"

        # Extract using time indices:
        t_all = df["t_s"].to_numpy(float)
        I = df["current_a"].to_numpy(float)
        V = df["voltage_v"].to_numpy(float)
        step = df["step_type"].astype(str).str.lower().to_numpy()
        rest_mask = (step == "rest") | (np.abs(I) <= 0.02)

        # find rest segments and take tail points
        refs_t, refs_soc = [], []
        start = None
        for i, m in enumerate(rest_mask):
            if m and start is None:
                start = i
            if (not m or i == len(rest_mask) - 1) and start is not None:
                end = i if not m else i + 1
                dur = t_all[end - 1] - t_all[start]
                if np.isfinite(dur) and dur >= 120.0:
                    t_end = t_all[end - 1]
                    tail_mask = (t_all[start:end] >= (t_end - 30.0))
                    if tail_mask.sum() >= 2:
                        v_med = float(np.nanmedian(V[start:end][tail_mask]))
                        soc_est = float(ocv_inv_fn(v_med))
                        refs_t.append(float(t_end))
                        refs_soc.append(clamp01(soc_est))
                start = None

        if len(refs_t) < 3:
            return None, None, "Too few rest-tail points for OCV inversion reference"

        return np.array(refs_t), np.array(refs_soc), "SOC ref from rest-tail OCV inversion"

    if ref_mode == "HPPC plateau SOC labels":
        soc_pct = df["soc_pct"].to_numpy(float)
        if not np.isfinite(soc_pct).any():
            return None, None, "No SOC/DOD(%) column to detect plateaus"
        step = df["step_type"].astype(str).str.lower().to_numpy()
        t_all = df["t_s"].to_numpy(float)

        rest_mask = step == "rest"
        refs_t, refs_soc = [], []
        start = None
        for i, m in enumerate(rest_mask):
            if m and start is None:
                start = i
            if (not m or i == len(rest_mask) - 1) and start is not None:
                end = i if not m else i + 1
                dur = t_all[end - 1] - t_all[start]
                if np.isfinite(dur) and dur >= 60.0:
                    soc_med = float(np.nanmedian(soc_pct[start:end])) / 100.0
                    refs_t.append(float(t_all[end - 1]))
                    refs_soc.append(clamp01(soc_med))
                start = None

        if len(refs_t) < 3:
            return None, None, "Too few plateaus detected"
        return np.array(refs_t), np.array(refs_soc), "Plateau SOC labels (rest medians)"

    return None, None, "Unknown reference mode"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Battery SOC estimation", layout="wide")
st.title("🔋 Battery SOC estimation (OCV + HPPC → 2RC EKF)")

st.markdown(
    """
Upload your **OCV Excel** and **HPPC Excel** (and optional **capacity test Excel**).
This app will:

- Build an **OCV(SOC)** curve robustly using Coulomb Counting.
- Run a **2RC + hysteresis EKF** on HPPC.
- Show **voltage fit**, **SOC estimate**, and **reference SOC comparisons** (cleanly).
"""
)

with st.sidebar:
    st.header("1) Upload files")
    ocv_file = st.file_uploader("OCV file (.xlsx)", type=["xlsx"])
    hppc_file = st.file_uploader("HPPC file (.xlsx)", type=["xlsx"])
    cap_file = st.file_uploader("Capacity test file (.xlsx) (optional)", type=["xlsx"])

    st.header("2) OCV building")
    bin_step = st.slider("OCV bin step (SOC)", 0.001, 0.02, 0.002, 0.001)
    rest_thresh = st.slider("Rest current threshold |I| (A)", 0.0, 0.2, 0.02, 0.005)
    min_rest_s = st.slider("Min rest duration (s)", 30, 600, 120, 30)
    tail_take_s = st.slider("Tail window (s)", 10, 120, 30, 5)

    st.header("3) EKF parameters")
    soc0 = st.slider("Initial SOC0", 0.0, 1.0, 0.80, 0.005)

    st.subheader("2RC parameters")
    R0 = st.number_input("R0 (ohm)", value=0.0100, format="%.6f")
    R1 = st.number_input("R1 (ohm)", value=0.00265, format="%.6f")
    C1 = st.number_input("C1 (F)", value=6775.5, format="%.2f")
    R2 = st.number_input("R2 (ohm)", value=0.00557, format="%.6f")
    C2 = st.number_input("C2 (F)", value=58727.4, format="%.2f")

    st.subheader("Hysteresis")
    k_h = st.number_input("k_h (V amplitude)", value=0.0, format="%.4f")
    tau_h = st.number_input("tau_h (s)", value=300.0, format="%.1f")

    st.subheader("Noise tuning")
    q_soc = st.number_input("Process noise q_soc", value=1e-6, format="%.1e")
    q_v = st.number_input("Process noise q_v (for v1/v2)", value=1e-4, format="%.1e")
    q_h = st.number_input("Process noise q_h (for hysteresis)", value=1e-5, format="%.1e")
    r_v = st.number_input("Measurement noise r_v (V^2)", value=2e-4, format="%.1e")

    st.header("4) Reference SOC logic")
    ref_mode = st.selectbox(
        "SOC reference to compare against",
        ["Capacity-based", "OCV-rest inversion (tails)", "HPPC plateau SOC labels", "SOC/DOD(%) column"],
        index=0,
    )

    run_btn = st.button("Run analysis", type="primary")


# -----------------------------
# Run
# -----------------------------
if not run_btn:
    st.info("Upload files and click **Run analysis**.")
    st.stop()

if ocv_file is None or hppc_file is None:
    st.error("Please upload both an OCV file and an HPPC file.")
    st.stop()

# Load data
try:
    df_ocv = load_record_any_excel(ocv_file)
    df_hppc = load_record_any_excel(hppc_file)
except Exception as e:
    st.exception(e)
    st.stop()

# Capacity estimate
Q_AH = None
if cap_file is not None:
    try:
        df_cap = load_record_any_excel(cap_file)
        Q_AH = estimate_capacity_ah(df_cap)
    except Exception:
        Q_AH = None

# fallback: use capacity info in HPPC if no cap file
if Q_AH is None:
    Q_AH = estimate_capacity_ah(df_hppc)

if Q_AH is None:
    Q_AH = 3.16  # Fallback tailored to your NMC data
    cap_note = "Capacity not found in uploaded files; using fallback Q_AH=3.16Ah"
else:
    cap_note = f"Estimated Q_AH={Q_AH:.4f}Ah from uploaded file(s)"

st.success(cap_note)

# Build OCV table
try:
    # Always extract using the robust Coulomb counting methodology
    soc_pts, v_pts = extract_rest_ocv_points(
        df_ocv,
        Q_AH=float(Q_AH),
        rest_thresh_a=rest_thresh,
        min_rest_s=float(min_rest_s),
        tail_take_s=float(tail_take_s),
    )
    if len(soc_pts) < 10:
        # fallback: try rest extraction from HPPC (often has many rests)
        soc_pts2, v_pts2 = extract_rest_ocv_points(
            df_hppc,
            Q_AH=float(Q_AH),
            rest_thresh_a=rest_thresh,
            min_rest_s=float(min_rest_s),
            tail_take_s=float(tail_take_s),
        )
        soc_pts = np.r_[soc_pts, soc_pts2]
        v_pts = np.r_[v_pts, v_pts2]

    if len(soc_pts) < 10:
        raise RuntimeError("Not enough rest-tail points to build OCV. Try lowering min_rest_s or increasing rest_thresh.")

    # bin & median
    soc_bin = np.round(soc_pts / bin_step) * bin_step
    ocv_tab = pd.DataFrame({"soc": soc_bin, "ocv_v": v_pts}).groupby("soc", as_index=False)["ocv_v"].median()
    ocv_tab = ocv_tab.sort_values("soc").reset_index(drop=True)

except Exception as e:
    st.error("Failed to build OCV curve.")
    st.exception(e)
    st.stop()

# Build OCV interpolation and derivative
try:
    ocv_fn = make_interp(ocv_tab["soc"], ocv_tab["ocv_v"], kind="linear")
    # derivative by finite difference table -> interp
    soc_arr = ocv_tab["soc"].to_numpy(float)
    v_arr = ocv_tab["ocv_v"].to_numpy(float)
    dv = np.gradient(v_arr, soc_arr)
    docv_fn = make_interp(soc_arr, dv, kind="linear")

    # inverse for OCV-rest reference
    ocv_inv_fn = enforce_monotonic_inverse(ocv_tab.rename(columns={"ocv_v": "ocv_v"}))
except Exception as e:
    st.error("Failed to build OCV interpolators.")
    st.exception(e)
    st.stop()

# Determine sign convention
sign_flip = detect_discharge_sign(df_hppc)

# Run EKF
res = run_ekf_2rc_h(
    df=df_hppc,
    ocv_fn=ocv_fn,
    docv_fn=docv_fn,
    R0=float(R0), R1=float(R1), C1=float(C1), R2=float(R2), C2=float(C2),
    Q_AH=float(Q_AH),
    soc0=float(soc0),
    k_h=float(k_h),
    tau_h=float(tau_h),
    q_soc=float(q_soc),
    q_v=float(q_v),
    q_h=float(q_h),
    r_v=float(r_v),
    sign_flip=int(sign_flip),
)

# Build reference SOC series
t_ref, soc_ref, ref_label = build_soc_reference(df_hppc, ref_mode, Q_AH, ocv_inv_fn=ocv_inv_fn)

# Compute SOC RMSE (if possible)
soc_rmse = None
if t_ref is not None and soc_ref is not None:
    # interpolate EKF SOC to reference timestamps
    soc_est_interp = np.interp(t_ref, res["t_s"], res["soc_est"])
    soc_rmse = float(np.sqrt(np.mean((soc_est_interp - soc_ref) ** 2)))

# Rest residual RMSE
# Measure how close rest voltage is to OCV(soc_est) (+ hysteresis)
dfh = df_hppc.copy()
step = dfh["step_type"].astype(str).str.lower().to_numpy()
rest_mask = (step == "rest") | (np.abs(dfh["current_a"].to_numpy(float)) <= rest_thresh)
V_meas = res["V_meas"]
I_int = res["I_int"]
soc_est = res["soc_est"]
h_est = res["h"]
V_ocv_only = np.array([float(ocv_fn(s)) for s in soc_est]) + h_est
rest_res = (V_meas - V_ocv_only)[rest_mask]
rest_rmse = float(np.sqrt(np.nanmean(rest_res ** 2))) if np.isfinite(rest_res).any() else np.nan

# -----------------------------
# Results + plots
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Voltage RMSE (V)", f"{res['rmse_v']:.6f}")
c2.metric("Rest residual RMSE (V)", f"{rest_rmse:.6f}" if np.isfinite(rest_rmse) else "N/A")
c3.metric("SOC RMSE", f"{soc_rmse:.6f}" if soc_rmse is not None else "N/A")

st.caption(f"sign_flip={res['sign_flip']} (discharge positive internally). Reference: {ref_label}")

# Plot OCV
st.subheader("OCV curve")
fig = plt.figure()
plt.plot(ocv_tab["soc"], ocv_tab["ocv_v"], marker="o")
plt.xlabel("SOC [0..1]")
plt.ylabel("OCV [V]")
plt.grid(True)
plt.title("OCV(SOC)")
st.pyplot(fig)

# Plot Voltage
st.subheader("HPPC voltage: measured vs EKF predicted")
t_min = res["t_s"] / 60.0
fig = plt.figure()
plt.plot(t_min, res["V_meas"], label="Measured V")
plt.plot(t_min, res["V_pred"], label="EKF Predicted V", linestyle="--")
plt.xlabel("Time [min]")
plt.ylabel("Voltage [V]")
plt.grid(True)
plt.legend()
st.pyplot(fig)

# Plot Current
st.subheader("HPPC current")
fig = plt.figure()
plt.plot(t_min, res["I_int"])
plt.xlabel("Time [min]")
plt.ylabel("Current (A) (discharge positive internal)")
plt.grid(True)
st.pyplot(fig)

# Plot SOC
st.subheader("SOC estimate (and reference if available)")
fig = plt.figure()
plt.plot(t_min, res["soc_est"], label="EKF SOC")
if t_ref is not None and soc_ref is not None:
    if len(t_ref) == len(t_min) and np.allclose(t_ref, res["t_s"]):
        plt.plot(t_ref / 60.0, soc_ref, label=f"Reference: {ref_mode}", linestyle="--")
    else:
        plt.scatter(t_ref / 60.0, soc_ref, s=20, label=f"Reference: {ref_mode}", color='red')
plt.xlabel("Time [min]")
plt.ylabel("SOC [0..1]")
plt.ylim([-0.05, 1.05])
plt.grid(True)
plt.legend()
st.pyplot(fig)

# Show parameters summary
st.subheader("Model summary")
st.write(
    {
        "Q_AH": float(Q_AH),
        "R0": float(R0), "R1": float(R1), "C1": float(C1), "R2": float(R2), "C2": float(C2),
        "k_h": float(k_h), "tau_h": float(tau_h),
        "q_soc": float(q_soc), "q_v": float(q_v), "q_h": float(q_h), "r_v": float(r_v),
        "sign_flip": int(sign_flip),
        "ref_mode": ref_mode,
    }
)

# Export results
st.subheader("Download results")
out = pd.DataFrame({
    "t_s": res["t_s"],
    "current_a_raw": res["I_raw"],
    "current_a_internal": res["I_int"],
    "voltage_meas_v": res["V_meas"],
    "voltage_pred_v": res["V_pred"],
    "soc_est": res["soc_est"],
    "v1": res["v1"],
    "v2": res["v2"],
    "h": res["h"],
})
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download EKF results CSV", data=csv_bytes, file_name="ekf_results.csv", mime="text/csv")

ocv_bytes = ocv_tab.to_csv(index=False).encode("utf-8")
st.download_button("Download OCV table CSV", data=ocv_bytes, file_name="ocv_table.csv", mime="text/csv")