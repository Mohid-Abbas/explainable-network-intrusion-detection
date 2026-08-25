import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, Any, List

st.set_page_config(page_title="Explainable Network Intrusion Detection", layout="wide")

st.title("Explainable Network Intrusion Detection Dashboard")
st.markdown("Upload or enter flow features to get a prediction with SHAP-based explanation.")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

sample_features = {
    "flow_duration": 128473,
    "total_fwd_packets": 12,
    "total_bwd_packets": 8,
    "total_length_of_fwd_packets": 1200,
    "total_length_of_bwd_packets": 800,
    "fwd_packet_length_max": 250,
    "fwd_packet_length_min": 40,
    "bwd_packet_length_max": 200,
    "bwd_packet_length_min": 40,
    "flow_bytes_per_sec": 15000,
    "flow_packets_per_sec": 150,
    "flow_iat_mean": 500,
    "flow_iat_std": 100,
    "flow_iat_max": 800,
    "flow_iat_min": 100,
    "fwd_iat_total": 3000,
    "fwd_iat_mean": 250,
    "fwd_iat_std": 50,
    "fwd_iat_max": 400,
    "fwd_iat_min": 50,
    "bwd_iat_total": 2000,
    "bwd_iat_mean": 250,
    "bwd_iat_std": 50,
    "bwd_iat_max": 350,
    "bwd_iat_min": 50,
    "fwd_psh_flags": 0,
    "bwd_psh_flags": 0,
    "fwd_urg_flags": 0,
    "bwd_urg_flags": 0,
    "fwd_header_length": 240,
    "bwd_header_length": 160,
    "fwd_packets_per_sec": 75,
    "bwd_packets_per_sec": 50,
    "min_packet_length": 40,
    "max_packet_length": 250,
    "packet_length_mean": 100,
    "packet_length_std": 30,
    "packet_length_variance": 900,
    "fin_flag_count": 0,
    "syn_flag_count": 1,
    "rst_flag_count": 0,
    "psh_flag_count": 0,
    "ack_flag_count": 1,
    "urg_flag_count": 0,
    "cwe_flag_count": 0,
    "ece_flag_count": 0,
    "down_up_ratio": 0.67,
    "average_packet_size": 100,
    "avg_fwd_segment_size": 100,
    "avg_bwd_segment_size": 100,
    "fwd_header_length_2": 240,
    "bwd_header_length_2": 160,
    "subflow_fwd_packets": 12,
    "subflow_fwd_bytes": 1200,
    "subflow_bwd_packets": 8,
    "subflow_bwd_bytes": 800,
    "init_win_bytes_forward": 8192,
    "init_win_bytes_backward": 8192,
    "act_data_pkt_fwd": 12,
    "min_seg_size_forward": 20,
    "active_mean": 1000,
    "active_std": 200,
    "active_max": 1200,
    "active_min": 800,
    "idle_mean": 5000,
    "idle_std": 1000,
    "idle_max": 6000,
    "idle_min": 4000,
}

mode = st.radio("Input Mode", ["Single Flow", "Batch CSV"], horizontal=True)

if mode == "Single Flow":
    st.subheader("Single Flow Prediction")
    with st.form("flow_form"):
        flow_duration = st.number_input("Flow Duration", value=128473)
        total_fwd_packets = st.number_input("Total Fwd Packets", value=12)
        total_bwd_packets = st.number_input("Total Bwd Packets", value=8)
        total_length_of_fwd_packets = st.number_input("Total Length of Fwd Packets", value=1200)
        total_length_of_bwd_packets = st.number_input("Total Length of Bwd Packets", value=800)
        fwd_packet_length_max = st.number_input("Fwd Packet Length Max", value=250)
        fwd_packet_length_min = st.number_input("Fwd Packet Length Min", value=40)
        bwd_packet_length_max = st.number_input("Bwd Packet Length Max", value=200)
        bwd_packet_length_min = st.number_input("Bwd Packet Length Min", value=40)
        flow_bytes_per_sec = st.number_input("Flow Bytes/s", value=15000)
        flow_packets_per_sec = st.number_input("Flow Packets/s", value=150)
        flow_iat_mean = st.number_input("Flow IAT Mean", value=500)
        flow_iat_std = st.number_input("Flow IAT Std", value=100)
        flow_iat_max = st.number_input("Flow IAT Max", value=800)
        flow_iat_min = st.number_input("Flow IAT Min", value=100)
        fwd_iat_total = st.number_input("Fwd IAT Total", value=3000)
        fwd_iat_mean = st.number_input("Fwd IAT Mean", value=250)
        fwd_iat_std = st.number_input("Fwd IAT Std", value=50)
        fwd_iat_max = st.number_input("Fwd IAT Max", value=400)
        fwd_iat_min = st.number_input("Fwd IAT Min", value=50)
        bwd_iat_total = st.number_input("Bwd IAT Total", value=2000)
        bwd_iat_mean = st.number_input("Bwd IAT Mean", value=250)
        bwd_iat_std = st.number_input("Bwd IAT Std", value=50)
        bwd_iat_max = st.number_input("Bwd IAT Max", value=350)
        bwd_iat_min = st.number_input("Bwd IAT Min", value=50)
        fwd_psh_flags = st.number_input("Fwd PSH Flags", value=0)
        bwd_psh_flags = st.number_input("Bwd PSH Flags", value=0)
        fwd_urg_flags = st.number_input("Fwd URG Flags", value=0)
        bwd_urg_flags = st.number_input("Bwd URG Flags", value=0)
        fwd_header_length = st.number_input("Fwd Header Length", value=240)
        bwd_header_length = st.number_input("Bwd Header Length", value=160)
        fwd_packets_per_sec = st.number_input("Fwd Packets/s", value=75)
        bwd_packets_per_sec = st.number_input("Bwd Packets/s", value=50)
        min_packet_length = st.number_input("Min Packet Length", value=40)
        max_packet_length = st.number_input("Max Packet Length", value=250)
        packet_length_mean = st.number_input("Packet Length Mean", value=100)
        packet_length_std = st.number_input("Packet Length Std", value=30)
        packet_length_variance = st.number_input("Packet Length Variance", value=900)
        fin_flag_count = st.number_input("FIN Flag Count", value=0)
        syn_flag_count = st.number_input("SYN Flag Count", value=1)
        rst_flag_count = st.number_input("RST Flag Count", value=0)
        psh_flag_count = st.number_input("PSH Flag Count", value=0)
        ack_flag_count = st.number_input("ACK Flag Count", value=1)
        urg_flag_count = st.number_input("URG Flag Count", value=0)
        cwe_flag_count = st.number_input("CWE Flag Count", value=0)
        ece_flag_count = st.number_input("ECE Flag Count", value=0)
        down_up_ratio = st.number_input("Down/Up Ratio", value=0.67)
        average_packet_size = st.number_input("Average Packet Size", value=100)
        avg_fwd_segment_size = st.number_input("Avg Fwd Segment Size", value=100)
        avg_bwd_segment_size = st.number_input("Avg Bwd Segment Size", value=100)
        fwd_header_length_2 = st.number_input("Fwd Header Length 2", value=240)
        bwd_header_length_2 = st.number_input("Bwd Header Length 2", value=160)
        subflow_fwd_packets = st.number_input("Subflow Fwd Packets", value=12)
        subflow_fwd_bytes = st.number_input("Subflow Fwd Bytes", value=1200)
        subflow_bwd_packets = st.number_input("Subflow Bwd Packets", value=8)
        subflow_bwd_bytes = st.number_input("Subflow Bwd Bytes", value=800)
        init_win_bytes_forward = st.number_input("Init Win Bytes Forward", value=8192)
        init_win_bytes_backward = st.number_input("Init Win Bytes Backward", value=8192)
        act_data_pkt_fwd = st.number_input("Act Data Pkt Fwd", value=12)
        min_seg_size_forward = st.number_input("Min Seg Size Forward", value=20)
        active_mean = st.number_input("Active Mean", value=1000)
        active_std = st.number_input("Active Std", value=200)
        active_max = st.number_input("Active Max", value=1200)
        active_min = st.number_input("Active Min", value=800)
        idle_mean = st.number_input("Idle Mean", value=5000)
        idle_std = st.number_input("Idle Std", value=1000)
        idle_max = st.number_input("Idle Max", value=6000)
        idle_min = st.number_input("Idle Min", value=4000)
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "features": {
                "flow_duration": flow_duration,
                "total_fwd_packets": total_fwd_packets,
                "total_bwd_packets": total_bwd_packets,
                "total_length_of_fwd_packets": total_length_of_fwd_packets,
                "total_length_of_bwd_packets": total_length_of_bwd_packets,
                "fwd_packet_length_max": fwd_packet_length_max,
                "fwd_packet_length_min": fwd_packet_length_min,
                "bwd_packet_length_max": bwd_packet_length_max,
                "bwd_packet_length_min": bwd_packet_length_min,
                "flow_bytes_per_sec": flow_bytes_per_sec,
                "flow_packets_per_sec": flow_packets_per_sec,
                "flow_iat_mean": flow_iat_mean,
                "flow_iat_std": flow_iat_std,
                "flow_iat_max": flow_iat_max,
                "flow_iat_min": flow_iat_min,
                "fwd_iat_total": fwd_iat_total,
                "fwd_iat_mean": fwd_iat_mean,
                "fwd_iat_std": fwd_iat_std,
                "fwd_iat_max": fwd_iat_max,
                "fwd_iat_min": fwd_iat_min,
                "bwd_iat_total": bwd_iat_total,
                "bwd_iat_mean": bwd_iat_mean,
                "bwd_iat_std": bwd_iat_std,
                "bwd_iat_max": bwd_iat_max,
                "bwd_iat_min": bwd_iat_min,
                "fwd_psh_flags": fwd_psh_flags,
                "bwd_psh_flags": bwd_psh_flags,
                "fwd_urg_flags": fwd_urg_flags,
                "bwd_urg_flags": bwd_urg_flags,
                "fwd_header_length": fwd_header_length,
                "bwd_header_length": bwd_header_length,
                "fwd_packets_per_sec": fwd_packets_per_sec,
                "bwd_packets_per_sec": bwd_packets_per_sec,
                "min_packet_length": min_packet_length,
                "max_packet_length": max_packet_length,
                "packet_length_mean": packet_length_mean,
                "packet_length_std": packet_length_std,
                "packet_length_variance": packet_length_variance,
                "fin_flag_count": fin_flag_count,
                "syn_flag_count": syn_flag_count,
                "rst_flag_count": rst_flag_count,
                "psh_flag_count": psh_flag_count,
                "ack_flag_count": ack_flag_count,
                "urg_flag_count": urg_flag_count,
                "cwe_flag_count": cwe_flag_count,
                "ece_flag_count": ece_flag_count,
                "down_up_ratio": down_up_ratio,
                "average_packet_size": average_packet_size,
                "avg_fwd_segment_size": avg_fwd_segment_size,
                "avg_bwd_segment_size": avg_bwd_segment_size,
                "fwd_header_length_2": fwd_header_length_2,
                "bwd_header_length_2": bwd_header_length_2,
                "subflow_fwd_packets": subflow_fwd_packets,
                "subflow_fwd_bytes": subflow_fwd_bytes,
                "subflow_bwd_packets": subflow_bwd_packets,
                "subflow_bwd_bytes": subflow_bwd_bytes,
                "init_win_bytes_forward": init_win_bytes_forward,
                "init_win_bytes_backward": init_win_bytes_backward,
                "act_data_pkt_fwd": act_data_pkt_fwd,
                "min_seg_size_forward": min_seg_size_forward,
                "active_mean": active_mean,
                "active_std": active_std,
                "active_max": active_max,
                "active_min": active_min,
                "idle_mean": idle_mean,
                "idle_std": idle_std,
                "idle_max": idle_max,
                "idle_min": idle_min,
            }
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            r.raise_for_status()
            result = r.json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            result = None

        if result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", result.get("prediction", "N/A"))
            with col2:
                st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
            with col3:
                st.metric("Malicious", "Yes" if result.get("is_malicious") else "No")
            st.subheader("Top Contributing Features (SHAP)")
            top_features = result.get("top_features", [])
            if top_features:
                feat_df = pd.DataFrame(top_features)
                st.bar_chart(feat_df.set_index("feature")["shap_value"])
            else:
                st.info("No SHAP explanation available for this prediction.")

elif mode == "Batch CSV":
    st.subheader("Batch CSV Scoring")
    uploaded_file = st.file_uploader("Upload CSV with flow features", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(df.head())
        if st.button("Run Batch Prediction"):
            flows = []
            for _, row in df.iterrows():
                flow = {k: (0.0 if pd.isna(v) else v) for k, v in row.to_dict().items()}
                flows.append(flow)
            try:
                r = requests.post(f"{API_URL}/predict/batch", json={"flows": flows}, timeout=60)
                r.raise_for_status()
                results = r.json()
            except Exception as e:
                st.error(f"Batch request failed: {e}")
                results = []
            if results:
                res_df = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(res_df)
                csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")
