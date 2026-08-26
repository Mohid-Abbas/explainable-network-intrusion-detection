from typing import Any

from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    flow_duration: float = Field(..., description="Duration of the flow in microseconds")
    total_fwd_packets: int = Field(..., description="Total packets in forward direction")
    total_bwd_packets: int = Field(..., description="Total packets in backward direction")
    total_length_of_fwd_packets: float = Field(..., description="Total bytes in forward direction")
    total_length_of_bwd_packets: float = Field(..., description="Total bytes in backward direction")
    fwd_packet_length_max: float = Field(..., description="Max packet length in forward direction")
    fwd_packet_length_min: float = Field(..., description="Min packet length in forward direction")
    bwd_packet_length_max: float = Field(..., description="Max packet length in backward direction")
    bwd_packet_length_min: float = Field(..., description="Min packet length in backward direction")
    flow_bytes_per_sec: float = Field(..., description="Flow bytes per second")
    flow_packets_per_sec: float = Field(..., description="Flow packets per second")
    flow_iat_mean: float = Field(..., description="Mean inter-arrival time between flows")
    flow_iat_std: float = Field(..., description="Std of inter-arrival time between flows")
    flow_iat_max: float = Field(..., description="Max inter-arrival time between flows")
    flow_iat_min: float = Field(..., description="Min inter-arrival time between flows")
    fwd_iat_total: float = Field(..., description="Total inter-arrival time between forward packets")
    fwd_iat_mean: float = Field(..., description="Mean inter-arrival time between forward packets")
    fwd_iat_std: float = Field(..., description="Std of inter-arrival time between forward packets")
    fwd_iat_max: float = Field(..., description="Max inter-arrival time between forward packets")
    fwd_iat_min: float = Field(..., description="Min inter-arrival time between forward packets")
    bwd_iat_total: float = Field(..., description="Total inter-arrival time between backward packets")
    bwd_iat_mean: float = Field(..., description="Mean inter-arrival time between backward packets")
    bwd_iat_std: float = Field(..., description="Std of inter-arrival time between backward packets")
    bwd_iat_max: float = Field(..., description="Max inter-arrival time between backward packets")
    bwd_iat_min: float = Field(..., description="Min inter-arrival time between backward packets")
    fwd_psh_flags: int = Field(..., description="Number of PSH flags in forward direction")
    bwd_psh_flags: int = Field(..., description="Number of PSH flags in backward direction")
    fwd_urg_flags: int = Field(..., description="Number of URG flags in forward direction")
    bwd_urg_flags: int = Field(..., description="Number of URG flags in backward direction")
    fwd_header_length: int = Field(..., description="Total bytes used for headers in forward direction")
    bwd_header_length: int = Field(..., description="Total bytes used for headers in backward direction")
    fwd_packets_per_sec: float = Field(..., description="Packets per second in forward direction")
    bwd_packets_per_sec: float = Field(..., description="Packets per second in backward direction")
    min_packet_length: float = Field(..., description="Minimum packet length")
    max_packet_length: float = Field(..., description="Maximum packet length")
    packet_length_mean: float = Field(..., description="Mean packet length")
    packet_length_std: float = Field(..., description="Std deviation of packet length")
    packet_length_variance: float = Field(..., description="Variance of packet length")
    fin_flag_count: int = Field(..., description="Number of FIN flags")
    syn_flag_count: int = Field(..., description="Number of SYN flags")
    rst_flag_count: int = Field(..., description="Number of RST flags")
    psh_flag_count: int = Field(..., description="Number of PSH flags")
    ack_flag_count: int = Field(..., description="Number of ACK flags")
    urg_flag_count: int = Field(..., description="Number of URG flags")
    cwe_flag_count: int = Field(..., description="Number of CWE flags")
    ece_flag_count: int = Field(..., description="Number of ECE flags")
    down_up_ratio: float = Field(..., description="Download and upload ratio")
    average_packet_size: float = Field(..., description="Average packet size")
    avg_fwd_segment_size: float = Field(..., description="Average segment size in forward direction")
    avg_bwd_segment_size: float = Field(..., description="Average segment size in backward direction")
    fwd_header_length_2: int = Field(..., description="Forward header length (duplicate for compatibility)")
    bwd_header_length_2: int = Field(..., description="Backward header length (duplicate for compatibility)")
    subflow_fwd_packets: float = Field(..., description="Average packets in a subflow in forward direction")
    subflow_fwd_bytes: float = Field(..., description="Average bytes in a subflow in forward direction")
    subflow_bwd_packets: float = Field(..., description="Average packets in a subflow in backward direction")
    subflow_bwd_bytes: float = Field(..., description="Average bytes in a subflow in backward direction")
    init_win_bytes_forward: int = Field(..., description="Total bytes with init win flag in forward direction")
    init_win_bytes_backward: int = Field(..., description="Total bytes with init win flag in backward direction")
    act_data_pkt_fwd: int = Field(..., description="Count of packets with at least 1 byte of TCP data payload in forward direction")
    min_seg_size_forward: float = Field(..., description="Minimum segment size in forward direction")
    active_mean: float = Field(..., description="Mean time a flow was active before becoming idle")
    active_std: float = Field(..., description="Std deviation of time a flow was active before becoming idle")
    active_max: float = Field(..., description="Maximum time a flow was active before becoming idle")
    active_min: float = Field(..., description="Minimum time a flow was active before becoming idle")
    idle_mean: float = Field(..., description="Mean time a flow was idle before becoming active")
    idle_std: float = Field(..., description="Std deviation of time a flow was idle before becoming active")
    idle_max: float = Field(..., description="Maximum time a flow was idle before becoming active")
    idle_min: float = Field(..., description="Minimum time a flow was idle before becoming active")

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    features: FlowFeatures


class BatchPredictRequest(BaseModel):
    flows: list[FlowFeatures]


class TopFeature(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    is_malicious: bool
    top_features: list[TopFeature]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    model_version: str
    training_date: str | None = None
    metrics_summary: dict[str, Any] | None = None

    model_config = {"protected_namespaces": ()}
