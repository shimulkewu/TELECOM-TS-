import json

# 1. Simulate a raw Telecom-TS observability record based on Feng et al. (2025)
raw_telemetry = {
    "metadata": {
        "timestamp": "2026-10-26T08:45:00.100Z",
        "node_id": "5G_GNB_TERENGGANU_04",
        "interface_index": "eth0.201"
    },
    "metrics": {
        "packet_loss_rate": 0.042,     # Float metric
        "jitter_ms": 12.8,             # Float metric
        "cpu_util_pct": 88.5,          # Float metric
        "link_status_bits": 1,         # Integer indicator
        "bgp_neighbor_state": "ACTIVE" # Categorical state
    },
    "label": "Core Link Flap"
}

# 2. Serialize using your explicit schema that preserves absolute scale variables
serialized_text = (
    f"[SCHEMA_START]\n"
    f"META | TS: {raw_telemetry['metadata']['timestamp']} | "
    f"NODE: {raw_telemetry['metadata']['node_id']} | "
    f"INT: {raw_telemetry['metadata']['interface_index']}\n"
    f"SYM  | LNK_BIT: {raw_telemetry['metrics']['link_status_bits']} | "
    f"BGP_STATE: {raw_telemetry['metrics']['bgp_neighbor_state']}\n"
    f"PERF | PKT_LOSS: {raw_telemetry['metrics']['packet_loss_rate']:.4f} | "
    f"JTR: {raw_telemetry['metrics']['jitter_ms']:.1f}ms | "
    f"CPU: {raw_telemetry['metrics']['cpu_util_pct']:.1f}%\n"
    f"TARGET | CLASS: {raw_telemetry['label']}\n"
    f"[SCHEMA_END]"
)

# 3. Print side-by-side style layout for your visual artifact
print("="*45 + " RAW TELEMETRY INPUT " + "="*45)
print(json.dumps(raw_telemetry, indent=4))
print("\n" + "="*43 + " PROPOSED SERIALIZED OUTPUT " + "="*43)
print(serialized_text)
print("="*111)
