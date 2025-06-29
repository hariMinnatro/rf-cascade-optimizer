
import streamlit as st
import pandas as pd
import os
import math
from itertools import product
from typing import List, Dict, Tuple

# === Conversion Utilities ===
def db_to_linear(db: float) -> float:
    return 10 ** (db / 10)

def linear_to_db(linear: float) -> float:
    return -math.inf if linear <= 0 else 10 * math.log10(linear)

def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10)

def mw_to_dbm(mw: float) -> float:
    return -math.inf if mw <= 0 else 10 * math.log10(mw)

# === Your Original cascade_analysis function with filter rejection ===
def cascade_analysis(stages: List[Dict], input_power_dbm: float, bw_hz: float = 1e6, temp_k: float = 290) -> Tuple[pd.DataFrame, Dict]:
    k = 1.38064852e-23
    thermal_noise_dbm = 10 * math.log10(k * temp_k * bw_hz) + 30

    gain_list_db = [s['gain_db'] for s in stages]
    nf_list_db = [s['nf_db'] for s in stages]
    p1db_list_dbm = [s.get('p1db_dbm', 100) for s in stages]
    iip3_list_dbm = [s.get('iip3_dbm', 100) for s in stages]
    rejection_list_db = [s.get('rejection_db', 0) for s in stages]

    gain_list_linear = [db_to_linear(g) for g in gain_list_db]
    nf_list_linear = [db_to_linear(nf) for nf in nf_list_db]
    p1db_list_mw = [dbm_to_mw(p) for p in p1db_list_dbm]
    iip3_list_mw = [dbm_to_mw(p) for p in iip3_list_dbm]
    rejection_list_linear = [db_to_linear(-r) for r in rejection_list_db]

    cumulative_gain_linear = 1.0
    total_cascaded_nf_linear = 0.0
    cumulative_iip3_inv_mw = 0.0
    cumulative_p1db_inv_mw = 0.0
    current_input_power_dbm = input_power_dbm

    for i in range(len(stages)):
        if i == 0:
            total_cascaded_nf_linear = nf_list_linear[0]
        else:
            total_cascaded_nf_linear += (nf_list_linear[i] - 1) / cumulative_gain_linear

        if i == 0:
            cumulative_iip3_inv_mw = 1 / iip3_list_mw[0]
        else:
            rejection_factor = 1.0
            for j in range(i):
                rejection_factor *= rejection_list_linear[j]
            cumulative_iip3_inv_mw += (cumulative_gain_linear * rejection_factor) / iip3_list_mw[i]

        if i == len(stages) - 1:
            cumulative_p1db_inv_mw += 1 / p1db_list_mw[i]
        else:
            cumulative_p1db_inv_mw += 1 / (p1db_list_mw[i] * sum(gain_list_linear[i+1:]))

        cumulative_gain_linear *= gain_list_linear[i]
        current_input_power_dbm += gain_list_db[i]

    total_gain_db = linear_to_db(cumulative_gain_linear)
    total_nf_db = linear_to_db(total_cascaded_nf_linear)
    total_p1db_dbm = mw_to_dbm(1 / cumulative_p1db_inv_mw if cumulative_p1db_inv_mw else math.inf)
    total_iip3_dbm = mw_to_dbm(1 / cumulative_iip3_inv_mw if cumulative_iip3_inv_mw else math.inf)

    output_power = min(input_power_dbm + total_gain_db, total_p1db_dbm + total_gain_db)
    noise_floor = thermal_noise_dbm + total_nf_db
    snr = output_power - noise_floor
    dr_p1db = (total_p1db_dbm + total_gain_db) - noise_floor
    dr_iip3 = (2 / 3) * ((total_iip3_dbm + total_gain_db) - noise_floor)

    return pd.DataFrame(), {
        "Total Gain (dB)": total_gain_db,
        "Total NF (dB)": total_nf_db,
        "Total IIP3 (dBm)": total_iip3_dbm,
        "Total P1dB (dBm)": total_p1db_dbm,
        "Output Power (dBm)": output_power,
        "Noise Floor (dBm)": noise_floor,
        "SNR (dB)": snr,
        "Dynamic Range (P1dB)": dr_p1db,
        "Dynamic Range (IIP3)": dr_iip3,
        "OP1dB (dBm)": total_p1db_dbm + total_gain_db,
        "OIP3 (dBm)": total_iip3_dbm + total_gain_db
    }

st.title("📡 RF Cascade Optimizer")

st.markdown("Upload multiple CSV files for different components (LNA, Filter, Attenuator, etc.)")
uploaded_files = st.file_uploader("Upload component CSV files", accept_multiple_files=True, type=["csv"])

if uploaded_files:
    component_db = {}
    for file in uploaded_files:
        name = file.name.replace(".csv", "")
        df = pd.read_csv(file)
        component_db[name] = df.to_dict("records")

    system_freq = st.number_input("System frequency (GHz)", value=2.4) * 1e9
    input_power = st.number_input("Input power (dBm)", value=-30.0)
    chain_input = st.text_input("Component Chain (e.g. LNA,[Filter],Attenuator)", "LNA,[Filter],Attenuator")

    target_spec = {}
    col1, col2 = st.columns(2)
    with col1:
        target_spec["min_gain_db"] = st.number_input("Min Gain (dB)", value=20.0)
        target_spec["min_p1db_dbm"] = st.number_input("Min P1dB (dBm)", value=10.0)
    with col2:
        target_spec["max_nf_db"] = st.number_input("Max NF (dB)", value=5.0)
        target_spec["min_iip3_dbm"] = st.number_input("Min IIP3 (dBm)", value=20.0)

    if st.button("Run Optimization"):
        sequence = []
        optional_flags = []
        for item in chain_input.split(","):
            item = item.strip()
            if item.startswith("[") and item.endswith("]"):
                sequence.append(item[1:-1])
                optional_flags.append(True)
            else:
                sequence.append(item)
                optional_flags.append(False)

        def filter_by_frequency(component_db, freq):
            filtered = {}
            for k, comps in component_db.items():
                valid = []
                for c in comps:
                    fmin = c.get("fmin", 0)
                    fmax = c.get("fmax", float("inf"))
                    if fmin <= freq <= fmax:
                        valid.append(c)
                if valid:
                    filtered[k] = valid
            return filtered

        component_db = filter_by_frequency(component_db, system_freq)

        from itertools import product

        KEY_MAPPING = {
            "gain_db": "Total Gain (dB)",
            "nf_db": "Total NF (dB)",
            "p1db_dbm": "Total P1dB (dBm)",
            "iip3_dbm": "Total IIP3 (dBm)"
        }

        def score_result(summary: Dict, target: Dict) -> float:
            score = 0.0
            for key, val in target.items():
                core = key.replace("min_", "").replace("max_", "")
                mapped = KEY_MAPPING.get(core)
                if not mapped or mapped not in summary: continue
                actual = summary[mapped]
                if key.startswith("min_"):
                    score += max(0, val - actual)
                elif key.startswith("max_"):
                    score += max(0, actual - val)
            return score

        def is_valid(summary: Dict, target: Dict) -> bool:
            for key, val in target.items():
                core = key.replace("min_", "").replace("max_", "")
                mapped = KEY_MAPPING.get(core)
                if not mapped: continue
                actual = summary[mapped]
                if key.startswith("min_") and actual < val: return False
                if key.startswith("max_") and actual > val: return False
            return True

        pools = []
        for comp, is_optional in zip(sequence, optional_flags):
            pool = component_db.get(comp, [])
            if is_optional:
                pool.append(None)
            pools.append(pool)

        results = []
        for combo in product(*pools):
            actual_chain = [c for c in combo if c]
            if not actual_chain:
                continue
            try:
                _, summary = cascade_analysis(actual_chain, input_power)
                summary["components"] = " → ".join(c["name"] for c in actual_chain)
                summary["score"] = score_result(summary, target_spec)
                if is_valid(summary, target_spec):
                    results.append(summary)
            except Exception as e:
                st.warning(f"Error in combo: {e}")

        results = sorted(results, key=lambda x: x["score"])[:5]

        if results:
            st.subheader("Top 5 Valid Combinations")
            for i, r in enumerate(results, 1):
                st.markdown(f"### Combination #{i}")
                st.write(f"**Chain:** {r['components']}")
                for k, v in r.items():
                    if k not in ["components", "score"]:
                        st.write(f"{k}: {v:.2f}")
                st.write(f"Score: {r['score']:.2f}")
        else:
            st.error("No valid combination found.")
