"""The four NF-v3 datasets and the 47 NetFlow v3 features used by every model.

All four datasets share one schema (the NetFlow v3 feature set of the NF-v3
release), so one feature list covers them. Address, port and timestamp
columns are never read: the CSV reader uses an explicit whitelist of the 47
behavioural features plus the ``Attack`` label column.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import paths as _paths

# Canonical name -> raw CSV column. Order defines the model input order.
NETFLOW_V3_ALIAS = {
    "protocol": "PROTOCOL",
    "l7_proto": "L7_PROTO",
    "in_bytes": "IN_BYTES",
    "out_bytes": "OUT_BYTES",
    "in_pkts": "IN_PKTS",
    "out_pkts": "OUT_PKTS",
    "tcp_flags": "TCP_FLAGS",
    "client_tcp_flags": "CLIENT_TCP_FLAGS",
    "server_tcp_flags": "SERVER_TCP_FLAGS",
    "flow_duration_ms": "FLOW_DURATION_MILLISECONDS",
    "duration_in": "DURATION_IN",
    "duration_out": "DURATION_OUT",
    "min_ttl": "MIN_TTL",
    "max_ttl": "MAX_TTL",
    "longest_flow_pkt": "LONGEST_FLOW_PKT",
    "shortest_flow_pkt": "SHORTEST_FLOW_PKT",
    "min_ip_pkt_len": "MIN_IP_PKT_LEN",
    "max_ip_pkt_len": "MAX_IP_PKT_LEN",
    "src_to_dst_second_bytes": "SRC_TO_DST_SECOND_BYTES",
    "dst_to_src_second_bytes": "DST_TO_SRC_SECOND_BYTES",
    "retransmitted_in_bytes": "RETRANSMITTED_IN_BYTES",
    "retransmitted_in_pkts": "RETRANSMITTED_IN_PKTS",
    "retransmitted_out_bytes": "RETRANSMITTED_OUT_BYTES",
    "retransmitted_out_pkts": "RETRANSMITTED_OUT_PKTS",
    "src_to_dst_avg_throughput": "SRC_TO_DST_AVG_THROUGHPUT",
    "dst_to_src_avg_throughput": "DST_TO_SRC_AVG_THROUGHPUT",
    "num_pkts_up_to_128_bytes": "NUM_PKTS_UP_TO_128_BYTES",
    "num_pkts_128_to_256_bytes": "NUM_PKTS_128_TO_256_BYTES",
    "num_pkts_256_to_512_bytes": "NUM_PKTS_256_TO_512_BYTES",
    "num_pkts_512_to_1024_bytes": "NUM_PKTS_512_TO_1024_BYTES",
    "num_pkts_1024_to_1514_bytes": "NUM_PKTS_1024_TO_1514_BYTES",
    "tcp_win_max_in": "TCP_WIN_MAX_IN",
    "tcp_win_max_out": "TCP_WIN_MAX_OUT",
    "icmp_type": "ICMP_TYPE",
    "icmp_ipv4_type": "ICMP_IPV4_TYPE",
    "dns_query_id": "DNS_QUERY_ID",
    "dns_query_type": "DNS_QUERY_TYPE",
    "dns_ttl_answer": "DNS_TTL_ANSWER",
    "ftp_command_ret_code": "FTP_COMMAND_RET_CODE",
    "src_to_dst_iat_min": "SRC_TO_DST_IAT_MIN",
    "src_to_dst_iat_max": "SRC_TO_DST_IAT_MAX",
    "src_to_dst_iat_avg": "SRC_TO_DST_IAT_AVG",
    "src_to_dst_iat_stddev": "SRC_TO_DST_IAT_STDDEV",
    "dst_to_src_iat_min": "DST_TO_SRC_IAT_MIN",
    "dst_to_src_iat_max": "DST_TO_SRC_IAT_MAX",
    "dst_to_src_iat_avg": "DST_TO_SRC_IAT_AVG",
    "dst_to_src_iat_stddev": "DST_TO_SRC_IAT_STDDEV",
}
FEATURE_COLUMNS: list[str] = list(NETFLOW_V3_ALIAS.values())
# Raw columns present in the CSVs that are never read (identity and time).
NETFLOW_V3_EXCLUDED = (
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
)
LABEL_COLUMN = "Attack"
BINARY_LABEL_COLUMN = "Label"
BENIGN_LABEL = "Benign"
EXCLUDE_LABEL = "__EXCLUDE__"

DATASET_NAMES = ("NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CICIDS2018-v3")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label_col: str = LABEL_COLUMN
    benign_label: str = BENIGN_LABEL
    kind: str = "file"

    @property
    def feature_alias(self) -> dict[str, str]:
        return NETFLOW_V3_ALIAS

    @property
    def path(self) -> str:
        """CSV location, resolved against DATA_ROOT at call time."""
        return _paths.dataset_csv_path(self.name)

    @property
    def paths(self) -> list[str]:
        return [self.path]


DATASET_REGISTRY: dict[str, DatasetSpec] = {name: DatasetSpec(name) for name in DATASET_NAMES}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]
