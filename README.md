# Signal Lah: GeoAI Connectivity Impact Simulator

**ASEAN GeoAI Fusion 2026 Hackathon**
Team: veronww2019 & jyng0204 | Theme: Consumer Empowerment

## Problem
Existing network maps show WHERE connectivity is poor.
Signal Lah asks: WHERE does poor connectivity matter MOST to communities?

## Pilot Area
Kuala Selangor, Selangor, Malaysia

## How to Run the Dashboard
1. Open the `web/` folder
2. Double-click `index.html` — done!

## Data Sources
- Network: Ookla Speedtest Open Data (gps_mobile_tiles)
- Boundaries: geoBoundaries (MYS ADM2)
- Population: WorldPop 2025 Malaysia 100m
- Schools & Healthcare: OpenStreetMap / curated

## GeoAI Method
1. Spatial aggregation → grid cells (~2km)
2. K-Means clustering → AI connectivity profiles
3. Scenario impact scoring → 0–100 priority score

## Team
- veronww2019 (Lead): Data & GeoAI
- jyng0204 (Member): Dashboard & presentation
