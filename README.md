# 📡 Signal Lah: GeoAI Connectivity Impact Simulator

**ASEAN GeoAI Fusion 2026 Hackathon**  
**Theme:** Consumer Empowerment  
**Pilot Area:** Kuala Selangor, Selangor, Malaysia

🌐 **Live Demo:** https://veronww2019-hub.github.io/SignahLah---MapLah/

---

## 1. Problem

Conventional connectivity maps are good at showing **where network performance is weak**, but they do not always show **where connectivity limitations matter most to communities**.

Signal Lah combines observed mobile-network performance with population and mapped essential services to help users identify areas where connectivity issues may have greater community impact.

The project focuses on three decision lenses:

- 🌐 **General Priority** — community exposure to connectivity gaps
- 📚 **Education Priority** — population and mapped-school exposure
- 🏥 **Healthcare Priority** — population and mapped-healthcare exposure

Signal Lah also separates **poor measured connectivity** from **missing evidence**, so areas without observed network measurements are not automatically treated as either good or bad coverage.

---

## 2. What Signal Lah Does

Signal Lah provides an interactive GeoAI decision-support dashboard that can:

- map relative connectivity-impact priority across Kuala Selangor;
- compare General, Education and Healthcare impact lenses;
- identify **Evidence Gaps** where no observed Ookla measurements are available;
- show learned GeoAI connectivity-community profiles;
- display mapped schools, healthcare facilities and locality context;
- rank the Top 5 priority cells for the selected impact lens;
- test hypothetical connectivity outcomes using download speed, upload speed and latency;
- compare **Before → After** priority scores for measured cells; and
- summarize district-wide effects under empirical Typical, Strong and Top observed scenarios.

---

## 3. Why GeoAI?

Signal Lah uses geospatial analysis to combine several location-based datasets into a common analysis grid, then applies machine learning to identify recurring connectivity-community patterns.

The project deliberately keeps **priority scoring transparent** rather than making priority a black-box AI prediction.

### GeoAI role

A K-Means model is trained only on cells with observed connectivity data using:

1. average download speed;
2. average upload speed;
3. average latency;
4. log-transformed population;
5. mapped-school count; and
6. mapped-healthcare count.

The final model uses **K = 3** and produces three descriptive profiles:

- **Strong-Connectivity Areas**
- **High-Exposure Essential-Service Hubs**
- **Connectivity-Constrained Areas**

Cells without observed network measurements are **not forced into a cluster**. They are labelled:

> **Unclassified – Evidence Gap**

This prevents the model from inventing a connectivity profile where supporting network evidence is unavailable.

---

## 4. Data Sources

| Dataset | Purpose |
|---|---|
| **Ookla Speedtest Open Data** | Observed mobile download, upload and latency measurements |
| **WorldPop 2025 Malaysia 100 m** | Estimated population exposure |
| **OpenStreetMap / Geofabrik** | Mapped schools, healthcare facilities and locality names |
| **geoBoundaries MYS ADM2** | Kuala Selangor district boundary |

### Final pilot dataset

- **331** analysis cells
- **182** cells with observed network measurements
- **149** Evidence Gap cells
- **8,441** network tests represented
- approximately **312,424** estimated population represented
- **105** OSM-mapped schools
- **64** OSM-mapped healthcare facilities
- **57** mapped localities used for geographic context

> Counts of schools, healthcare facilities and localities refer to features mapped in OpenStreetMap/Geofabrik and should not be interpreted as complete official administrative registers.

---

## 5. Data Processing Pipeline

```text
Raw geospatial datasets
        │
        ├── Kuala Selangor ADM2 boundary
        ├── Ookla mobile-performance tiles
        ├── WorldPop population raster
        └── OpenStreetMap facilities + localities
        │
        ▼
2 km analysis grid
        │
        ▼
Spatial aggregation
        ├── test-weighted network metrics
        ├── population exposure
        ├── school counts
        └── healthcare counts
        │
        ▼
Transparent impact scoring
        ├── General
        ├── Education
        └── Healthcare
        │
        ├───────────────┐
        ▼               ▼
Evidence handling    K-Means GeoAI
        │               │
        └──────┬────────┘
               ▼
      Interactive GeoJSON
               │
               ▼
      Leaflet web dashboard
               │
               ▼
      What-if scenario engine
```

The preprocessing pipeline is reproducible through the Python scripts in `scripts/`.

---

## 6. Transparent Impact Scoring

Signal Lah does **not** use AI to directly assign priority.

For cells with observed network measurements:

1. network performance is converted into empirical percentile-based connectivity-gap components;
2. population and mapped essential services are converted into exposure measures;
3. connectivity gap and exposure are combined into separate General, Education and Healthcare impact scores; and
4. relative priority tiers are derived from the Kuala Selangor score distribution.

Priority classes are:

- **High**
- **Medium**
- **Low**
- **Evidence Gap**

### Important

These priority categories are **relative within the Kuala Selangor pilot**.

They are **not MCMC regulatory service classifications**, universal broadband standards, or guarantees of service quality.

---

## 7. Evidence-Aware Decision Support

A major design principle of Signal Lah is:

> **No observed measurement ≠ good connectivity, and no observed measurement ≠ poor connectivity.**

Cells without observed Ookla measurements are displayed separately as **Evidence Gaps**.

For measured cells, Signal Lah also keeps evidence strength separate from impact priority. This allows a high-priority area with limited supporting observations to be flagged for additional validation rather than presented with false certainty.

Examples of decision cues include:

- **High priority – supported**
- **High priority – validate evidence**
- **Evidence Gap**

---

## 8. What-If Connectivity Simulator

Signal Lah includes a browser-based connectivity-outcome simulator.

For a selected measured cell, users can enter hypothetical:

- download speed;
- upload speed; and
- latency.

The simulator then recalculates:

- connectivity-gap score;
- General priority score;
- Education priority score;
- Healthcare priority score;
- priority tier; and
- GeoAI profile under the hypothetical scenario.

It also includes empirical presets derived from the observed Kuala Selangor network distribution:

- **Typical**
- **Strong**
- **Top**

An **Upgrade-only** safeguard prevents a proposed intervention from unintentionally worsening an already-better measured metric.

### Simulator scope

The simulator answers:

> **“What if this area achieved these connectivity outcomes?”**

It does **not** predict tower placement, RF propagation, signal strength, infrastructure cost, or guarantee that a specific telecommunications intervention will achieve the entered speeds.

---

## 9. District-Wide What-If Analysis

Signal Lah also summarizes the effect of empirical connectivity scenarios across measured cells in Kuala Selangor.

The dashboard can show:

- high-priority cells before the scenario;
- high-priority cells after the scenario;
- number of high-priority cells moving out of High;
- mean impact-score reduction; and
- estimated population and mapped services located within cells whose modelled priority changes.

> Population and facility counts in changed cells are **contextual exposure counts**, not guaranteed beneficiary counts.

---

## 10. GeoAI Validation

The final K-Means model was evaluated across candidate values of `K = 2...6` using:

- silhouette score;
- Calinski-Harabasz score;
- Davies-Bouldin score; and
- stability across multiple random seeds.

`K = 3` was selected because it provided highly stable assignments and interpretable cluster sizes while avoiding very small clusters.

The three resulting profiles are used as **descriptive connectivity-community archetypes**, not as ground-truth classifications.

Further model details are documented in:

`docs/geoai_model_report.md`

---

## 11. Responsible GeoAI & Limitations

Signal Lah is designed as a **decision-support prototype**, not an automated infrastructure-allocation system.

Key limitations include:

- Ookla Speedtest observations are crowdsourced and do not represent complete network coverage;
- Evidence Gap cells must not be interpreted as having good or poor connectivity without further measurements;
- OSM-mapped schools and healthcare facilities may be incomplete;
- WorldPop values are modelled population estimates;
- priority thresholds are relative to the Kuala Selangor pilot;
- K-Means profiles are descriptive clusters rather than factual labels;
- one profile may contain transitional/mixed cells, so cluster assignments should be interpreted cautiously;
- simulator outputs represent hypothetical connectivity outcomes rather than RF/tower engineering predictions; and
- expanding Signal Lah to another district should include local recalibration rather than blindly reusing Kuala Selangor thresholds.

Human review and additional field/network evidence should be used before making real infrastructure decisions.

---

## 12. Technology Stack

### Geospatial / Data

- Python
- GeoPandas
- Pyogrio
- NumPy
- Pandas
- tifffile / imagecodecs / zarr

### GeoAI

- scikit-learn
- K-Means
- StandardScaler
- joblib

### Frontend

- HTML
- CSS
- Vanilla JavaScript
- Leaflet.js
- CARTO / OpenStreetMap basemap

### Deployment

- GitHub
- GitHub Actions
- GitHub Pages

---

## 13. Repository Structure

```text
SignahLah---MapLah/
│
├── data/
│   ├── processed/
│   │   ├── signal_lah_cells_v2.geojson
│   │   ├── signal_lah_cells_v2_enriched.geojson
│   │   └── scenario_presets.json
│   └── raw/                      # ignored from Git
│
├── docs/
│   ├── data_audit.md
│   └── geoai_model_report.md
│
├── models/
│   └── geoai_kmeans_k3.joblib
│
├── scripts/
│   ├── build_grid.py
│   ├── aggregate_ookla.py
│   ├── aggregate_population.py
│   ├── aggregate_schools.py
│   ├── aggregate_healthcare.py
│   ├── build_baseline_scores.py
│   ├── build_priority_labels.py
│   ├── train_geoai.py
│   ├── validate_geoai.py
│   ├── simulator_core.py
│   ├── simulate_intervention.py
│   ├── enrich_web_geography.py
│   └── ...
│
├── web/
│   ├── index.html
│   ├── signal_lah_cells_v2.geojson
│   ├── simulator_config.json
│   ├── scenario_presets.json
│   ├── schools.geojson
│   ├── healthcare.geojson
│   └── localities.geojson
│
├── .github/
│   └── workflows/
│       └── deploy-pages.yml
│
└── README.md
```

---

## 14. Run Locally

Do **not** open `web/index.html` directly using a `file://` URL because the dashboard fetches local GeoJSON/JSON assets.

From the repository root:

```bash
cd web
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/
```

Alternatively, use the deployed version:

🌐 https://veronww2019-hub.github.io/SignahLah---MapLah/

---

## 15. Suggested Demo Flow

A quick way to explore Signal Lah:

1. open **General Priority**;
2. inspect the **Top 5 Areas**;
3. search for a grid/locality;
4. compare General, Education and Healthcare priority;
5. turn on mapped schools and healthcare facilities;
6. open **Evidence Gaps** to identify where network evidence is missing;
7. open **GeoAI Profiles** to view learned area archetypes;
8. select a measured cell and run a What-If connectivity scenario; and
9. use **District What-If** to compare empirical connectivity scenarios across the pilot area.

---

## 16. Scalability

Kuala Selangor is used as a pilot area, but the workflow is designed to be transferable.

A new district can be processed by:

1. loading the target administrative boundary;
2. aggregating equivalent network, population and facility datasets;
3. rebuilding the local analysis grid;
4. recalculating local empirical impact thresholds;
5. retraining/revalidating GeoAI profiles where appropriate; and
6. exporting the resulting dataset to the same lightweight Leaflet interface.

The goal is a reusable **geospatial decision-support workflow**, not a Kuala Selangor-only static map.

---

## 17. AI-Assisted Development Disclosure

Generative AI tools, primarily **ChatGPT**, were used as development aids during the project for:

- code drafting, debugging and troubleshooting;
- guidance on geospatial data-processing and analysis workflows;
- assistance with documentation and technical explanations; and
- preparation and refinement of presentation and submission content.

AI-assisted outputs incorporated into the final project were **reviewed, tested and modified by the team before inclusion**. Final responsibility for the project design, dataset selection, GeoAI methodology, interpretation of results, validation, prototype behaviour and submission decisions remained with the team.

The project's K-Means GeoAI model is a separately implemented machine-learning component trained on the project's processed geospatial dataset; generative AI was used as a development assistant rather than as an autonomous decision-maker within the deployed Signal Lah dashboard.

---

## 18. Team

- **veronww2019** — Lead
- **jyng0204** — Member

Built for the **ASEAN GeoAI Fusion 2026 Hackathon — Consumer Empowerment**.
