\# Signal Lah Data Audit



\## Study Area

Kuala Selangor, Selangor, Malaysia.



Boundary source:

geoBoundaries MYS ADM2



\- Feature: Kuala Selangor

\- CRS: EPSG:4326

\- Approximate district area: 1,186.1 km²

\- Existing Signal Lah grid: 348 clipped cells



\## Ookla Mobile Performance



Raw source:

2026-01-01\_performance\_mobile\_tiles



Raw fields:

\- quadkey

\- avg\_d\_kbps

\- avg\_u\_kbps

\- avg\_lat\_ms

\- tests

\- devices



Audit results:

\- 916 tiles in Kuala Selangor bounding box

\- 757 tiles intersect district

\- 733 tile centres inside district

\- 8,441 total tests



Existing processed Signal Lah dataset:

\- 8,441 total tests



Conclusion:

The existing connectivity dataset is strongly reproducible from the raw

Ookla source. The raw source will be retained, but preprocessing will be

rebuilt as reproducible code.



\## Population



Raw source:

mys\_pop\_2025\_CN\_100m\_R2025A\_v1.tif



Properties:

\- EPSG:4326

\- approximately 100 m resolution

\- NoData = -99999



Audit:

\- Raw population assigned using pixel centres: 312,424.25

\- Existing processed population: 340,910.75

\- Cell-level correlation: 0.9977

\- Median absolute cell difference: 3.66



Conclusion:

The existing population layer almost certainly follows the same spatial

population source, but the aggregation cannot be reproduced exactly and

appears to overcount population in some cells.



The new pipeline will assign each population pixel to only one grid cell.



\## Education Facilities



Current OSM extract:

malaysia-singapore-brunei-260806-free.shp.zip



Within Kuala Selangor:

\- school points: 10

\- school polygons: 98

\- school records before deduplication: 108

\- broader education records before deduplication: 163



Existing processed dataset:

\- school\_count total: 8



A reconstruction using unique named school points also produced 8 records,

but their spatial grid assignments did not reproduce the existing dataset.



Conclusion:

The source/method used for the existing school\_count cannot be reliably

reproduced. Education facilities will be rebuilt from the current OSM

point and polygon layers with explicit deduplication.



\## Healthcare Facilities



Current OSM extract within Kuala Selangor:



Points:

\- clinics: 50

\- doctors: 1



Areas:

\- clinics: 10

\- hospitals: 3



Total before deduplication: 64



Existing processed dataset:

\- healthcare\_count total: 7



Only a small subset of the old healthcare grid locations correspond to

the current OSM healthcare records.



Conclusion:

The old healthcare layer cannot be reliably reproduced from the current

source and will be rebuilt.



\## Legacy AI and Scoring



The existing GeoJSON contains:

\- connectivity\_quality

\- connectivity\_gap

\- impact scores

\- priority classes

\- cluster

\- ai\_profile

\- data\_confidence



However, the repository does not contain a reproducible preprocessing

pipeline, trained model, scaler, clustering validation, or model artifact.



Conclusion:

Legacy AI labels and impact scores will be treated as reference/demo

outputs only.



The new project will rebuild:

1\. geospatial preprocessing,

2\. deterministic impact scoring,

3\. data-confidence handling,

4\. GeoAI model training and validation,

5\. scenario simulation.



\## Final Data Decisions



KEEP RAW SOURCES:

\- geoBoundaries Kuala Selangor boundary

\- Ookla mobile-performance tiles

\- 2025 population raster

\- current OSM POI point and polygon layers



REBUILD:

\- analysis grid

\- Ookla-to-grid aggregation

\- population-to-grid aggregation

\- education facility dataset

\- healthcare facility dataset

\- impact scores

\- data confidence

\- AI profiles

\- simulator outputs

