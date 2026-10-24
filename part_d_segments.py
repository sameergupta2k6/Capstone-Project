"""Part D (segments): hub profile, K-Means segmentation, separation evidence.

Run:  python part_d_segments.py
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

CLUSTER_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEPARATION_KS = [2, 3, 4, 5]
SEGMENT_ORDER = ["Standard", "Watch", "Escalate"]
TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def main():
    modelling = pd.read_csv(os.path.join(DATA, "modelling_table.csv"))
    hubs = pd.read_csv(os.path.join(DATA, "hubs.csv"))

    profile = build_hub_profile(modelling)
    profile.round(4).to_csv(os.path.join(OUTPUTS, "hub_profile.csv"), index=False)

    scaled = StandardScaler().fit_transform(profile[CLUSTER_COLS])

    write_separation(scaled)
    profile = assign_segments(profile, scaled)
    write_segment_outputs(profile, hubs)

    print("Part D (segments) complete:",
          profile["segment"].value_counts().reindex(SEGMENT_ORDER).to_dict())


def build_hub_profile(modelling):
    profile = (
        modelling.groupby("hub_id")
        .agg(
            shipments=("hub_id", "size"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    return profile


def write_separation(scaled):
    dmat = pairwise_distances(scaled)
    pairs = list(combinations(range(scaled.shape[0]), 2))
    rows = []
    for k in SEPARATION_KS:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        across = [dmat[i, j] for i, j in pairs if labels[i] != labels[j]]
        within = [dmat[i, j] for i, j in pairs if labels[i] == labels[j]]
        ratio = float(np.mean(across) / np.mean(within))
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        rows.append({
            "K": k,
            "across_over_within": round(ratio, 4),
            "cluster_sizes": "-".join(str(s) for s in sizes),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS, "part_d_separation.csv"), index=False)


def assign_segments(profile, scaled):
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    profile = profile.copy()
    profile["cluster"] = labels
    order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    cluster_to_name = {cluster: SEGMENT_ORDER[i] for i, cluster in enumerate(order)}
    profile["segment"] = profile["cluster"].map(cluster_to_name)
    return profile


def write_segment_outputs(profile, hubs):
    hub_segments = (
        profile[["hub_id", "segment"]]
        .merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
        [["hub_id", "segment", "hub_city", "hub_tier"]]
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    hub_segments.to_csv(os.path.join(OUTPUTS, "hub_segments.csv"), index=False)

    seg_profile = (
        profile.groupby("segment")
        .agg(
            hubs=("hub_id", "size"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reindex(SEGMENT_ORDER)
        .reset_index()
    )
    seg_profile[CLUSTER_COLS] = seg_profile[CLUSTER_COLS].round(4)
    seg_profile.to_csv(os.path.join(OUTPUTS, "segment_profile.csv"), index=False)

    expected = hub_segments["hub_tier"].map(TIER_TO_SEGMENT)
    mismatch = hub_segments[hub_segments["segment"] != expected].reset_index(drop=True)
    mismatch.to_csv(os.path.join(OUTPUTS, "tier_vs_segment_mismatch.csv"), index=False)


if __name__ == "__main__":
    main()
