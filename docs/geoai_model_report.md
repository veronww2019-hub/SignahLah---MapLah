# Signal Lah GeoAI Model Report

## Purpose

The GeoAI component uses unsupervised K-Means clustering to discover recurring connectivity-community profiles among analysis cells with observed mobile-network data. The model does not determine intervention priority; priority is produced separately by the transparent connectivity-impact model.

## Training Data

- Total analysis cells: 331
- Cells used for GeoAI training: 182
- Evidence-gap cells excluded from training: 149
- No network values were imputed for evidence-gap cells.

## Features

- `avg_download_mbps`
- `avg_upload_mbps`
- `avg_latency_ms`
- `log_population`
- `school_count`
- `healthcare_count`

All model features were standardized before K-Means training.

## Model Selection

Candidate cluster counts k=2 through k=6 were compared using silhouette score, Calinski-Harabasz score, Davies-Bouldin score, cluster size and repeated-seed Adjusted Rand Index stability.

The selected model uses k=3. Its mean stability ARI was 0.9919 and minimum stability ARI was 0.9798. The k=4 to k=6 alternatives produced very small clusters, while k=3 retained meaningful cluster sizes and extremely stable assignments.

## Final Internal Validation

- Silhouette score: 0.3013
- Calinski-Harabasz score: 64.62
- Davies-Bouldin score: 1.1704

## GeoAI Profiles

| Cluster | Profile | Cells | Mean silhouette | Median download (Mbps) | Median upload (Mbps) | Median latency (ms) | Median population |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Strong-Connectivity Areas | 70 | 0.304 | 196.67 | 30.04 | 19.31 | 1480.54 |
| 1 | High-Exposure Essential-Service Hubs | 19 | 0.095 | 157.28 | 24.53 | 23.08 | 4477.57 |
| 2 | Connectivity-Constrained Areas | 93 | 0.342 | 39.21 | 6.76 | 25.78 | 552.53 |

## Interpretation

- **Strong-Connectivity Areas:** comparatively strong download/upload performance, lower latency and lower connectivity-gap scores.
- **Dense Essential-Service Hubs:** high population and concentrations of mapped schools and healthcare facilities. This cluster is more mixed and should be interpreted as a transitional recurring profile rather than a sharply separated class.
- **Connectivity-Constrained Areas:** comparatively lower download/upload performance and higher connectivity-gap scores, typically with lower population and fewer mapped essential-service facilities.

## Important Limitations

- K-Means clusters are descriptive profiles, not ground-truth labels.
- Cluster IDs have no inherent ranking or severity meaning.
- The Dense Essential-Service Hubs cluster has weaker internal separation than the other two profiles.
- The model is trained only on cells with observed Ookla data.
- Absence of Ookla measurements is treated as an evidence gap, not proof of poor or good connectivity.
- OSM facility counts represent mapped facilities and may not equal official administrative totals.
- GeoAI profiles supplement rather than replace the transparent General, Education and Healthcare priority scores.