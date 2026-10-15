# EUDR-CH: a benchmark for verifying "deforestation-free" coffee in Vietnam's Central Highlands

**Version 0.9 (October 2026), without labels.** Code, documentation and the legal rule: <https://github.com/datrich/vEUDR>

The EU Deforestation Regulation (EUDR, Regulation (EU) 2023/1115) admits coffee only from land that was not deforested
after 31 December 2020. Verifying this needs two facts per plot: what the land was used for on that date, and whether
forest was converted to agriculture afterwards. EUDR-CH gives, for a stratified sample of 1780 forest-loss alert
patches and coffee plots in the Central Highlands of Vietnam, the satellite evidence needed to establish those facts:
Sentinel-2 reference sheets, per-scene Sentinel-2, Landsat and Sentinel-1 records, rendered single-date scenes,
outlines in geographic and image coordinates, and features from open map products. It is the dataset of the paper
*Facts First, Law in Code: Verifying "Deforestation-Free" Coffee under the EU Deforestation Regulation in Vietnam's
Central Highlands, with an Open Benchmark* (submitted).

Human reference labels (blind and model-assisted, with adjudications) will be added in version 1.0.

## Sample

| Kind | Items | Design |
|---|---|---|
| Alert patches | 1200 | Global Forest Watch integrated alerts (GLAD-L, GLAD-S2, RADD) grouped into patches of 8-connected pixels alerted within 90 days; first alert 2021-2024, >= 0.3 ha; 100 patches drawn at random in each of 12 strata (year x size 0.3-0.5 / 0.5-1 / >= 1 ha) |
| 1-ha plots | 580 | Squares centred on points with a Forest Data Partnership coffee probability >= 0.7 in 2024; strata from GFC2020 and tree-cover loss: B1 forest 2020 with loss 2021-2025 (224), B2 forest 2020 without loss (217), B3 not forest without loss (139). Plots are not cadastral parcels |

Provinces as before the July 2025 merger: Kon Tum, Gia Lai, Dak Lak, Dak Nong, Lam Dong. DEV/TEST split by 0.2 degree
blocks drawn with a fixed seed before any label existed: 38 of 125 blocks are DEV (395 items), the rest TEST (1385).
Use DEV only for anything that learns from labels.

## Files

| Path | Content |
|---|---|
| `items.csv` | One row per item: `item_id` (A... alerts, P... plots), `kind`, `stratum`, `stratum_population` (alerts: patches in the stratum), `n_sampled_in_stratum`, `wave`, `lon`, `lat` (centroid), `area_ha`, `first_alert_date`, `province_2020`, `block_0p2deg`, `split`, `has_scene_series` |
| `outlines.geojson` | Item polygons (WGS 84) |
| `map_features.parquet` | 66 features per item from open map products (shares of the item area unless stated); see `map_features_dictionary.csv` for each column's description, source and licence |
| `ndvi_s2_l1c_series.parquet` | Sentinel-2 L1C NDVI in the outline, clear scenes (>= 50 % clear pixels), one value per day, since 2016; 1729 items |
| `sentinel2_scenes.parquet` | Every Sentinel-2 scene over the item box since 2016: date, scene and tile, cloud share, frame coverage, clear share in the box and in the outline, NDVI in the outline (1.28 million rows, 1000 items with `has_scene_series`) |
| `landsat_scenes.parquet` | Every Landsat 5/7/8/9 scene since 2000, same fields (1.07 million rows) |
| `sentinel1_scenes.parquet` | Every Sentinel-1 IW scene since 2015: orbit pass, relative orbit, VV and VH backscatter in the outline (dB) (0.71 million rows) |
| `sheets/interpreter/<item>.png` | Interpreter sheet: 10 Sentinel-2 median composites (true colour and false-colour NIR) on a ~1 km box, outline in yellow, with the NDVI series. Alerts: dry 2019, dry 2020, cut-off (11/2020-04/2021), 4-1 months before the alert, 0-2, 3-6, 10-14 months, 2 and 3 years after, dry 2026. Plots: dry seasons 2017-2020, cut-off, dry seasons 2022-2026 |
| `sheets/standard/<item>.png` | Standard sheet (the one the vision-language model of the paper reads): alerts 6 windows from 12 months before to 20 months after the alert on a ~800 m box; plots 8 windows (dry 2017-2020, cut-off, 2022, 2024, 2026) on a ~600 m box |
| `sheets/layout_<kind>.json` | Per item: box (west, south, east, north, degrees), tile size, window labels and dates, outline rings in tile pixels |
| `scenes/<item>/` | Rendered single-date Sentinel-2 scenes (192 px JPEG, true colour `_tc` and false colour `_fc`) on the interpreter box: up to three clear scenes per year plus the scenes nearest the cut-off and the first alert; `scenes.csv` gives date, scene, clear shares, NDVI and why each was chosen |

### Pixel geometry of sheets and scenes

Each image tile covers the item centroid +- 0.0045 degrees (interpreter sheets and scenes; ~1 km, ~5 m per pixel) or
+- 0.0036 / 0.0027 degrees (standard sheets of alerts / plots), in 192 x 192 pixels. A point (lon, lat) maps to
`x = (lon - west) / (east - west) * 192`, `y = (north - lat) / (north - south) * 192` in a tile. On a sheet, the tile
of row `r` (0 true colour, 1 false colour) and column `c` starts at `(c * 192, r * 210 + 18)`; the NDVI chart takes
the bottom 240 pixels. The outline rings of the interpreter layout were checked against the yellow outline drawn by Earth Engine on the
tiles.

## Licence and sources

The dataset is released under **CC BY 4.0**. It is derived from:

- Copernicus Sentinel-1 and Sentinel-2 data (modified), processed in Google Earth Engine: free, attribution
  "contains modified Copernicus Sentinel data [2015-2026]";
- Landsat 5/7/8/9 Collection 2 (USGS, public domain);
- Global Forest Change v1.13 (Hansen et al.), JRC GFC2020 (versions 1-4), JRC Tropical Moist Forests, GFW integrated
  alerts, Dynamic World, ESA WorldCover 2020, ETH and Meta/WRI canopy-height maps, drivers of forest loss at 1 km
  (Sims et al. 2025): CC BY 4.0;
- NASA MODIS burned area and FIRMS (open data), Copernicus DEM GLO-30 (free, attribution).

Not included in this version: high-resolution imagery (Google Earth, Esri Wayback; viewed by annotators, not
redistributed), the Esri Wayback capture metadata, the official 2020 forest-status map of Vietnam and its lot
attributes, Forest Data Partnership and Nature Trace layers, WDPA, and the GLAD and JRC forest-type layers, whose
redistribution terms are being checked; human and model labels (version 1.0).

## Using the data

```python
from veudr.data import Dataset
from veudr import baselines, timeseries
ds = Dataset("eudr-ch")
overlays = baselines.map_overlays(ds.map_features)       # GFC2020 + loss, TMF, Dynamic World, drivers map
ts = timeseries.features(ds)                              # Sentinel-2 / Landsat / Sentinel-1 features per item
```

The legal rule (`veudr.legal_rule.apply_rule`) turns factual answers (land use in 2020, change after it) into the
EUDR conclusion; the guide for collecting such labels is `docs/LABELLING.md` in the code repository.

## Citation

Please cite the paper (reference to be added on publication) and this dataset:
EUDR-CH: a benchmark for verifying deforestation-free coffee in Vietnam's Central Highlands, version 0.9, 2026.
