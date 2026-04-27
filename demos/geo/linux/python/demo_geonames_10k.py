#!/usr/bin/env python3
# nn20db-sdk
#
# Copyright (c) 2026 Bruno Keymolen
# Contact: bruno.keymolen@gmail.com
#
# License:
# This SDK, including all pre-compiled binaries and accompanying files,
# is provided for private and educational use only.
#
# Commercial use is strictly prohibited without prior written agreement
# from the author.
#
# Disclaimer:
# This software is provided "as is", without any express or implied
# warranties, including but not limited to the implied warranties of
# merchantability and fitness for a particular purpose.
#
# In no event shall the author be held liable for any damages arising
# from the use of this software.

"""
demo_geonames_10k.py  —  nn20db GeoNames nearest-neighbour demo
================================================================
Downloads the GeoNames cities15000 dataset on first run, builds
an nn20db index of 10,000 cities using 3D unit-sphere vectors
derived from latitude/longitude, then runs example geo queries.

Nearest neighbours on the unit sphere under L2 are equivalent in
ordering to cosine similarity (for unit vectors |a|=|b|=1,
||a-b||^2 = 2 - 2*cos(θ), so smaller L2 == larger cosine).
We use the cosine_f32 metric here as it is the most natural choice
for unit-sphere vectors.

Usage (from repo root):
    python demos/geo/linux/python/demo_geonames_10k.py
    python demos/geo/linux/python/demo_geonames_10k.py --rebuild
    python demos/geo/linux/python/demo_geonames_10k.py --max-cities 5000 --top-k 5
"""

import sys
import os
import math
import struct
import zipfile
import urllib.request
import csv
import argparse
import time
import shutil
from pathlib import Path

# ── locate repo root and add api/python to the import path ──────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    p for p in (HERE, *HERE.parents)
    if (p / "api" / "python" / "nn20db.py").exists()
)
sys.path.insert(0, str(REPO_ROOT / "api" / "python"))

from nn20db import (
    NN20Db, DatabaseConfig, LfsStorageConfig, HnswConfig, HnswLevelConfig,
    CacheConfig,
)

# ── constants ────────────────────────────────────────────────────────────────
MAX_CITIES      = 10_000
TOP_K           = 10
EARTH_RADIUS_KM = 6371.0088

DATA_DIR    = HERE / "data"
DATASET_ZIP = DATA_DIR / "cities15000.zip"
DATASET_TXT = DATA_DIR / "cities15000.txt"
DB_PATH     = DATA_DIR / "geo10k"

# Fixed-length city metadata packed into each index entry:
#   32s  name (UTF-8, null-padded/truncated)
#    2s  country code
#     f  latitude  (float32)
#     f  longitude (float32)
#     i  population (int32)
CITY_META_FMT  = "<32s2sffi"
CITY_META_SIZE = struct.calcsize(CITY_META_FMT)   # 46 bytes

DOWNLOAD_URL = "https://download.geonames.org/export/dump/cities15000.zip"

# Demo queries: (label, lat, lon)
QUERIES = [
    ("Brussels, Belgium", 50.85045,   4.34878),
    ("Paris, France",     48.85660,   2.35220),
    ("New York, USA",     40.71280, -74.00600),
    ("Tokyo, Japan",      35.67620, 139.65030),
]


# ── vector helpers ────────────────────────────────────────────────────────────

def latlon_to_unit_vector(lat_deg: float, lon_deg: float) -> list:
    """Convert lat/lon (degrees) to a 3D unit vector on the unit sphere."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return [
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ── dataset download / parse ──────────────────────────────────────────────────

def ensure_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_TXT.exists():
        if not DATASET_ZIP.exists():
            print(f"Downloading GeoNames cities15000 from {DOWNLOAD_URL} ...")
            urllib.request.urlretrieve(DOWNLOAD_URL, DATASET_ZIP)
            print(f"  -> saved to {DATASET_ZIP}")
        print("Extracting cities15000.txt ...")
        with zipfile.ZipFile(DATASET_ZIP) as zf:
            zf.extract("cities15000.txt", DATA_DIR)
        print(f"  -> extracted to {DATASET_TXT}")
    else:
        print(f"Dataset already present at {DATASET_TXT}")


def load_cities(max_cities: int = MAX_CITIES) -> list:
    """
    Parse the GeoNames TSV and return a list of tuples:
        (name, lat, lon, country_code, population)
    Only the first max_cities valid rows are returned.
    """
    print(f"Loading up to {max_cities} cities ...")
    cities = []
    with open(DATASET_TXT, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(cities) >= max_cities:
                break
            try:
                name    = row[1]
                lat     = float(row[4])
                lon     = float(row[5])
                country = row[8]
                pop     = int(row[14]) if row[14].strip() else 0
            except (IndexError, ValueError):
                continue
            cities.append((name, lat, lon, country, pop))
    print(f"  -> loaded {len(cities)} cities")
    return cities


# ── database build / open ─────────────────────────────────────────────────────

def build_index(cities: list, db_path: Path, rebuild: bool = False) -> NN20Db:
    """
    Create a new nn20db index from *cities*, or open an existing one.
    When rebuild=True the existing index is deleted first.
    Returns an open NN20Db instance (caller must close or use as context manager).
    """
    if db_path.exists() and rebuild:
        print(f"Removing existing index at {db_path} ...")
        shutil.rmtree(db_path)

    if db_path.exists():
        print(f"Opening existing nn20db index at {db_path} ...")
        db = NN20Db.open(str(db_path))
        return db

    print(f"Creating nn20db index at {db_path} ...")
    db_path.mkdir(parents=True, exist_ok=True)

    cfg = DatabaseConfig(
        dimension=3,
        metadata_size=CITY_META_SIZE,
        metric="cosine",
        storage=LfsStorageConfig(
            device_path=str(db_path),
            lane_cache_size_kb=8192,
            lane_size_mb=128,
            log_size_mb=512,
            log_index_buckets=65536,
            object_cache_size_bytes=4096,
            read_ahead_size_bytes=4096,
            block_size=4096,
            disable_crc=True,
        ),
        index=HnswConfig(
            ef_search=64,
            max_levels=5,
            diversity_alpha=1.2,
            search_threads=1,
            search_seen_set_capacity=20000,
            levels=[
                HnswLevelConfig(M=32, ef_construction=120),
                HnswLevelConfig(M=16, ef_construction=80),
                HnswLevelConfig(M=8,  ef_construction=60),
                HnswLevelConfig(M=4,  ef_construction=30),
                HnswLevelConfig(M=2,  ef_construction=15),
            ],
        ),
        cache=CacheConfig(enabled=True, max_entries=1000),
    )

    db = NN20Db.create(cfg)

    print(f"Inserting {len(cities)} vectors ...")
    t0 = time.time()
    for idx, (name, lat, lon, country, pop) in enumerate(cities):
        vec      = latlon_to_unit_vector(lat, lon)
        metadata = struct.pack(
            CITY_META_FMT,
            name.encode("utf-8"),
            country.encode("utf-8"),
            lat,
            lon,
            pop,
        )
        db.add(vec, metadata=metadata)
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{len(cities)}  ({elapsed:.1f}s)")

    print("Syncing ...")
    db.sync()
    print(f"  -> indexing done in {time.time() - t0:.1f}s")
    return db


# ── queries ───────────────────────────────────────────────────────────────────

def run_queries(db: NN20Db, top_k: int = TOP_K) -> None:
    for query_label, qlat, qlon in QUERIES:
        qvec    = latlon_to_unit_vector(qlat, qlon)
        results = db.search_ef(qvec, k=top_k, ef_search=100)

        print(f"\nQuery: {query_label} ({qlat}, {qlon})")
        print(f"Top {top_k} nearest cities:")

        for rank, sr in enumerate(results, 1):
            _vec, meta = db.get(sr.id, dimension=3, metadata_size=CITY_META_SIZE)
            name_b, country_b, clat, clon, pop = struct.unpack(CITY_META_FMT, meta)
            name    = name_b.rstrip(b"\x00").decode("utf-8", errors="replace")
            country = country_b.rstrip(b"\x00").decode("utf-8", errors="replace")
            dist_km = haversine_km(qlat, qlon, clat, clon)
            print(
                f"  {rank:2d}. {name:<24s} {country}  "
                f"pop={pop:<10d}  "
                f"lat={clat:9.5f} lon={clon:9.5f}  "
                f"distance={dist_km:8.1f} km"
            )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="nn20db GeoNames 10k nearest-neighbour demo"
    )
    parser.add_argument(
        "--max-cities", type=int, default=MAX_CITIES,
        help=f"Number of cities to index (default: {MAX_CITIES})",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of nearest neighbours to return (default: {TOP_K})",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete and rebuild the index even if it already exists",
    )
    args = parser.parse_args()

    ensure_dataset()
    need_cities = not DB_PATH.exists() or args.rebuild
    cities = load_cities(args.max_cities) if need_cities else []
    with build_index(cities, DB_PATH, rebuild=args.rebuild) as db:
        run_queries(db, top_k=args.top_k)


if __name__ == "__main__":
    main()
