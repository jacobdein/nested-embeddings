"""AOI, source paths, and cache locations. Everything else reads from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# --- OlmoEarth v1.3 annual embeddings (Ai2), Source Cooperative --------------
OE_STORE_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
OE_ENDPOINT = "https://data.source.coop"
OE_ZONE = "utm30"  # EPSG:32630
OE_YEAR = 2024
CRS = "EPSG:32630"

# The zone grid: x0 = 0, y0 = 9_502_720, 10 m pixels (spatial:transform attr).
GRID_X0, GRID_Y0, RES = 0, 9_502_720, 10
SHARD = 2048

# AOI = exactly one 2048 x 2048 shard: shard row 185, shard col 33 of utm30.
# Centre 51.454 N, -0.322 E (3.9 km NW of Richmond Park centre); east edge at
# -0.169 E, so the whole square is west of 0 deg and inside UTM 30N.
SHARD_ROW, SHARD_COL = 185, 33
ROW0, COL0 = SHARD_ROW * SHARD, SHARD_COL * SHARD  # zone pixel offsets
N = SHARD  # AOI side in pixels
AOI_X0 = GRID_X0 + COL0 * RES  # 675_840 m
AOI_Y1 = GRID_Y0 - ROW0 * RES  # 5_713_920 m (top edge)
AOI_X1 = AOI_X0 + N * RES
AOI_Y0 = AOI_Y1 - N * RES
AOI_TRANSFORM = (RES, 0.0, AOI_X0, 0.0, -RES, AOI_Y1)  # affine (a,b,c,d,e,f)

N_LEVELS = 12  # 10 m ... 20.48 km; level 12 is the single AOI cell (= grand mean)
QUERY_LEVELS = list(range(1, 11))  # 10 m ... 5.12 km

# --- Reference labels and imagery (scoring and display only) -----------------
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N51W003_Map.tif"
)  # = s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51W003_Map.tif
# Sentinel-2 L2A collection 1 true colour, Element 84 Earth Search (AWS). Tile 30UXC
# covers the whole AOI; 2024-05-20 is the cloud-free 2024 date over it (the July
# scenes have cumulus over the south of the AOI).
S2_TCI_URLS = [
    "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/"
    "sentinel-2-c1-l2a/30/U/XC/2024/5/S2B_T30UXC_20240520T111145_L2A/TCI.tif",
]

EMB_CACHE = DATA / f"olmoearth_{OE_ZONE}_r{SHARD_ROW}_c{SHARD_COL}_{OE_YEAR}_int8.npy"
WC_CACHE = DATA / "worldcover_2021_v200_aoi.npy"
S2_CACHE = DATA / "s2_tci_2024_aoi.npy"
PYRAMID_DIR = DATA / "pyramid"
