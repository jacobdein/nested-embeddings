"""Download the AOI once and cache it under data/.

- OlmoEarth v1.3 embeddings, year 2024, all 128 bands, as raw int8 (C, H, W).
- ESA WorldCover 2021 v200, nearest-neighbour onto the embedding grid.
- Sentinel-2 L2A true-colour (TCI) for July 2024, onto the same grid. Display only.
"""

from __future__ import annotations

import json
import time

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject

from . import config as C


def open_oe_root():
    import zarr
    from obstore.store import S3Store

    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store.from_url(C.OE_STORE_URL, endpoint=C.OE_ENDPOINT, skip_signature=True)
    return zarr.open_group(zarr.storage.ObjectStore(s3, read_only=True), mode="r")


def quantization_attrs() -> dict:
    """The geoemb:quantization block of the zone group, as stored."""
    cache = C.DATA / "quantization_attrs.json"
    if cache.exists():
        return json.loads(cache.read_text())
    zone = open_oe_root()[C.OE_ZONE]
    attrs = dict(zone.attrs)
    out = {
        "geoemb:quantization": attrs["geoemb:quantization"],
        "proj:code": attrs["proj:code"],
        "spatial:transform": attrs["spatial:transform"],
        "geoemb:model": attrs["geoemb:model"],
        "geoemb:build_version": attrs["geoemb:build_version"],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def fetch_embeddings() -> np.ndarray:
    """(128, 2048, 2048) int8 for the AOI shard and year."""
    if C.EMB_CACHE.exists():
        return np.load(C.EMB_CACHE)
    zone = open_oe_root()[C.OE_ZONE]
    a, _, c, _, e, f = zone.attrs["spatial:transform"]
    assert (a, c, e, f) == (C.RES, C.GRID_X0, -C.RES, C.GRID_Y0), "zone grid changed"
    t = int(np.where(zone["time"][:] == C.OE_YEAR)[0][0])
    t0 = time.time()
    q = zone["embeddings"][t, :, C.ROW0 : C.ROW0 + C.N, C.COL0 : C.COL0 + C.N]
    print(f"  embeddings read in {time.time() - t0:.0f} s, shape {q.shape}")
    C.EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(C.EMB_CACHE, q)
    return q


def _warp_to_aoi(src_url: str, dst: np.ndarray, resampling=Resampling.nearest, src_nodata=None):
    with rasterio.open(src_url) as src:
        for b in range(dst.shape[0]):
            reproject(
                source=rasterio.band(src, b + 1),
                destination=dst[b],
                dst_transform=Affine(*C.AOI_TRANSFORM),
                dst_crs=C.CRS,
                resampling=resampling,
                src_nodata=src_nodata,
                dst_nodata=0,
                init_dest_nodata=False,
            )


def fetch_worldcover() -> np.ndarray:
    """(2048, 2048) uint8 WorldCover class codes on the embedding grid."""
    if C.WC_CACHE.exists():
        return np.load(C.WC_CACHE)
    dst = np.zeros((1, C.N, C.N), np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        _warp_to_aoi("/vsicurl/" + C.WORLDCOVER_URL, dst)
    np.save(C.WC_CACHE, dst[0])
    return dst[0]


def fetch_s2() -> np.ndarray:
    """(2048, 2048, 3) uint8 Sentinel-2 true colour, two MGRS tiles mosaicked."""
    if C.S2_CACHE.exists():
        return np.load(C.S2_CACHE)
    dst = np.zeros((3, C.N, C.N), np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for url in C.S2_TCI_URLS:  # later tiles only fill pixels still empty
            tile = np.zeros_like(dst)
            _warp_to_aoi("/vsicurl/" + url, tile, src_nodata=0)
            empty = dst.sum(0) == 0
            dst[:, empty] = tile[:, empty]
    rgb = np.moveaxis(dst, 0, -1)
    np.save(C.S2_CACHE, rgb)
    return rgb


def fetch_all():
    print("fetch: quantization attrs", quantization_attrs()["geoemb:quantization"]["formula"])
    print("fetch: embeddings", fetch_embeddings().shape)
    print("fetch: worldcover", fetch_worldcover().shape)
    print("fetch: sentinel-2", fetch_s2().shape)


if __name__ == "__main__":
    fetch_all()
