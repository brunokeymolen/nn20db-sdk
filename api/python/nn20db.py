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
nn20db Python API wrapper
=========================
Thin ctypes binding over the nn20db C SDK shared library.

Usage
-----
Build the shared library first::

    make -C api/python

Then::

    from python.api.nn20db import NN20Db, DatabaseConfig, LfsStorageConfig, HnswConfig

    cfg = DatabaseConfig(
        dimension=128,
        metadata_size=4,
        metric="euclidean",
        storage=LfsStorageConfig(device_path="/data/mydb"),
        index=HnswConfig(ef_search=64),
    )
    with NN20Db.create(cfg) as db:
        db.add([0.1, 0.2, ...], metadata=b"\\x01\\x00\\x00\\x00")
        results = db.search([0.1, 0.2, ...], k=10)
        for r in results:
            print(r.id, r.distance)
"""

import ctypes
import os
import struct
import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Locate and load the shared library
# ---------------------------------------------------------------------------

def _find_lib() -> str:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    candidates = [
        here / "libnn20db_py.so",
        repo_root / "build" / "libnn20db_py.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(
        "libnn20db_py.so not found. Run `make -C api/python` first."
    )


_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_find_lib())
        _bind_functions(_lib)
    return _lib


# ---------------------------------------------------------------------------
# Low-level ctypes structures  (all packed, matching C __attribute__((packed)))
# ---------------------------------------------------------------------------

class _StorageMemory(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("segments",     ctypes.c_uint16),
        ("segment_size", ctypes.c_uint32),
        ("device_path",  ctypes.c_char * 512),
        ("mount_point",  ctypes.c_char * 64),
        ("flags",        ctypes.c_uint16),
        ("reserved",     ctypes.c_uint8 * 6),
    ]


class _StorageLfs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("device_path",             ctypes.c_char * 512),
        ("mount_point",             ctypes.c_char * 64),
        ("lane_cache_size_kb",      ctypes.c_uint32),
        ("lane_size_mb",            ctypes.c_uint32),
        ("log_size_mb",             ctypes.c_uint32),
        ("log_index_buckets",       ctypes.c_uint32),
        ("object_cache_size_bytes", ctypes.c_uint32),
        ("read_ahead_size_bytes",   ctypes.c_uint32),
        ("block_size",              ctypes.c_uint32),
        ("flags",                   ctypes.c_uint16),
    ]


class _StorageUnion(ctypes.Union):
    _pack_ = 1
    _fields_ = [
        ("memory", _StorageMemory),
        ("lfs",    _StorageLfs),
    ]


class _StorageCacheConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("enabled",               ctypes.c_uint8),
        ("max_entries",           ctypes.c_uint32),
        ("max_object_size_bytes", ctypes.c_uint32),
    ]


class _StorageConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type",  ctypes.c_int),
        ("union", _StorageUnion),
        ("cache", _StorageCacheConfig),
    ]


class _IndexHnswLevel(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("M",               ctypes.c_uint16),
        ("ef_construction", ctypes.c_uint16),
    ]


class _IndexHnswConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("search_threads",           ctypes.c_uint16),
        ("max_levels",               ctypes.c_uint8),
        ("diversity_alpha",          ctypes.c_float),
        ("search_seen_set_capacity", ctypes.c_uint32),
        ("ef_search",                ctypes.c_uint16),
        ("level_config",             _IndexHnswLevel * 10),
    ]


class _IndexLinear(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("reserved", ctypes.c_int)]


class _IndexUnion(ctypes.Union):
    _pack_ = 1
    _fields_ = [
        ("linear", _IndexLinear),
        ("hnsw",   _IndexHnswConfig),
    ]


class _IndexConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type",  ctypes.c_int),
        ("union", _IndexUnion),
    ]


class _VectorConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type",          ctypes.c_int),
        ("dimension",     ctypes.c_int),
        ("metadata_size", ctypes.c_int),
    ]


class _MetricUnion(ctypes.Union):
    _pack_ = 1
    _fields_ = [("reserved", ctypes.c_int)]


class _MetricConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type",  ctypes.c_int),
        ("union", _MetricUnion),
    ]


class _LoggerConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("level",   ctypes.c_int),
        ("path",    ctypes.c_char * 128),
    ]


class _Config(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version_major", ctypes.c_uint16),
        ("version_minor", ctypes.c_uint16),
        ("version_patch", ctypes.c_uint16),
        ("vector",        _VectorConfig),
        ("storage",       _StorageConfig),
        ("metric",        _MetricConfig),
        ("index",         _IndexConfig),
        ("logger",        _LoggerConfig),
    ]


class _SearchResult(ctypes.Structure):
    # NOT packed — matches C struct without __attribute__((packed))
    # sizeof = 16: uint64(8) + float(4) + 4 bytes trailing padding
    _fields_ = [
        ("id",       ctypes.c_uint64),
        ("distance", ctypes.c_float),
    ]


class _DbListResult(ctypes.Structure):
    _fields_ = [
        ("path",          ctypes.c_char * 512),
        ("version_major", ctypes.c_int),
        ("version_minor", ctypes.c_int),
        ("version_patch", ctypes.c_int),
        ("index_type",    ctypes.c_int),
        ("metric_type",   ctypes.c_int),
        ("vector_type",   ctypes.c_int),
    ]


# ---------------------------------------------------------------------------
# Enum constants (mirrors C enums)
# ---------------------------------------------------------------------------

# nn20db_storage_type
STORAGE_MEMORY      = 0
STORAGE_BLOCKDEVICE = 1
STORAGE_FILE        = 2
STORAGE_LFS         = 3

# nn20db_index_type
INDEX_LINEAR = 0
INDEX_HNSW   = 1

# nn20db_dimension_type
DIM_FLOAT32 = 0
DIM_BIT     = 1

# nn20db_metric_type
METRIC_COSINE          = 0
METRIC_EUCLIDEAN       = 1
METRIC_EUCLIDEAN_AVX2  = 2
METRIC_DOT_PRODUCT     = 3
METRIC_MANHATTAN       = 4
METRIC_HAMMING         = 5
METRIC_JACCARD         = 6
METRIC_COSINE_F32      = 7
METRIC_EUCLIDEAN_F32   = 8

# error codes
ERROR_OK                       =  0
ERROR_INVALID_ARGUMENT         = -1
ERROR_OUT_OF_MEMORY            = -2
ERROR_NOT_FOUND                = -3
ERROR_IO                       = -4
ERROR_INTERNAL                 = -5
ERROR_NOT_IMPLEMENTED          = -6
ERROR_ALREADY_EXISTS           = -7
ERROR_STORAGE_FULL             = -8
ERROR_CORRUPTED_DATA           = -9
ERROR_CHECKSUM_MISMATCH        = -10
ERROR_END                      = -11
ERROR_DELETED                  = -12
ERROR_INDEX_HNSW_NO_FREE_EDGE_SLOTS = -13
ERROR_INDEX_HEAP_DISCARDED     = -14
ERROR_EMPTY                    = -15
ERROR_NOT_EXIST                = -16
ERROR_DB_ALREADY_EXISTS        = -17

_ERROR_NAMES = {v: k for k, v in globals().items() if k.startswith("ERROR_")}

_METRIC_NAMES = {
    "cosine":         METRIC_COSINE,
    "euclidean":      METRIC_EUCLIDEAN,
    "euclidean_avx2": METRIC_EUCLIDEAN_AVX2,
    "dot_product":    METRIC_DOT_PRODUCT,
    "manhattan":      METRIC_MANHATTAN,
    "hamming":        METRIC_HAMMING,
    "jaccard":        METRIC_JACCARD,
    "cosine_f32":     METRIC_COSINE_F32,
    "euclidean_f32":  METRIC_EUCLIDEAN_F32,
}


# ---------------------------------------------------------------------------
# Function binding
# ---------------------------------------------------------------------------

def _bind_functions(lib: ctypes.CDLL) -> None:
    lib.nn20db_create.argtypes = [ctypes.POINTER(_Config), ctypes.POINTER(ctypes.c_void_p)]
    lib.nn20db_create.restype  = ctypes.c_int

    lib.nn20db_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.nn20db_open.restype  = ctypes.c_int

    lib.nn20db_open_with_config.argtypes = [ctypes.POINTER(_Config), ctypes.POINTER(ctypes.c_void_p)]
    lib.nn20db_open_with_config.restype  = ctypes.c_int

    lib.nn20db_close.argtypes = [ctypes.c_void_p]
    lib.nn20db_close.restype  = ctypes.c_int

    lib.nn20db_dtor.argtypes = [ctypes.c_void_p]
    lib.nn20db_dtor.restype  = None

    lib.nn20db_vector_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.nn20db_vector_add.restype  = ctypes.c_int

    lib.nn20db_vector_search.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(_SearchResult),
    ]
    lib.nn20db_vector_search.restype = ctypes.c_int

    lib.nn20db_vector_search_ef.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_SearchResult),
    ]
    lib.nn20db_vector_search_ef.restype = ctypes.c_int

    lib.nn20db_vector_get.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.nn20db_vector_get.restype = ctypes.c_int

    lib.nn20db_vector_remove.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.nn20db_vector_remove.restype  = ctypes.c_int

    lib.nn20db_sync.argtypes = [ctypes.c_void_p]
    lib.nn20db_sync.restype  = ctypes.c_int

    lib.nn20db_compact.argtypes = [ctypes.c_void_p]
    lib.nn20db_compact.restype  = ctypes.c_int

    lib.nn20db_list_databases.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(_DbListResult)),
    ]
    lib.nn20db_list_databases.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class Nn20dbError(Exception):
    def __init__(self, rc: int, msg: str = ""):
        name = _ERROR_NAMES.get(rc, f"rc={rc}")
        super().__init__(f"nn20db error {name}: {msg}" if msg else f"nn20db error {name}")
        self.rc = rc


def _check(rc: int, op: str = "") -> None:
    if rc != ERROR_OK:
        raise Nn20dbError(rc, op)


# ---------------------------------------------------------------------------
# Python-level config dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HnswLevelConfig:
    M: int = 16
    ef_construction: int = 200


@dataclasses.dataclass
class HnswConfig:
    ef_search: int = 64
    max_levels: int = 5
    diversity_alpha: float = 1.2
    search_threads: int = 1
    search_seen_set_capacity: int = 0
    levels: List[HnswLevelConfig] = dataclasses.field(default_factory=lambda: [
        HnswLevelConfig(M=32, ef_construction=400),
        HnswLevelConfig(M=16, ef_construction=120),
        HnswLevelConfig(M=8,  ef_construction=60),
        HnswLevelConfig(M=4,  ef_construction=30),
        HnswLevelConfig(M=2,  ef_construction=15),
    ])


@dataclasses.dataclass
class LfsStorageConfig:
    device_path: str = ""
    mount_point: str = ""
    lane_cache_size_kb: int = 32 * 256
    lane_size_mb: int = 128
    log_size_mb: int = 1
    log_index_buckets: int = 512
    object_cache_size_bytes: int = 4096
    read_ahead_size_bytes: int = 4096
    block_size: int = 4096
    disable_crc: bool = False


@dataclasses.dataclass
class MemoryStorageConfig:
    segments: int = 1
    segment_size: int = 0


@dataclasses.dataclass
class CacheConfig:
    enabled: bool = False
    max_entries: int = 256
    max_object_size_bytes: int = 0


@dataclasses.dataclass
class DatabaseConfig:
    dimension: int = 0
    metadata_size: int = 0
    vector_type: str = "float32"   # "float32" or "bit"
    metric: str = "euclidean"
    storage: Union[LfsStorageConfig, MemoryStorageConfig] = dataclasses.field(
        default_factory=LfsStorageConfig
    )
    index: HnswConfig = dataclasses.field(default_factory=HnswConfig)
    cache: CacheConfig = dataclasses.field(default_factory=CacheConfig)


# ---------------------------------------------------------------------------
# Config conversion: Python dataclass -> ctypes _Config
# ---------------------------------------------------------------------------

def _to_c_config(cfg: DatabaseConfig) -> _Config:
    c = _Config()

    # vector
    c.vector.type          = DIM_BIT if cfg.vector_type == "bit" else DIM_FLOAT32
    c.vector.dimension     = cfg.dimension
    c.vector.metadata_size = cfg.metadata_size

    # metric
    metric_id = _METRIC_NAMES.get(cfg.metric)
    if metric_id is None:
        raise ValueError(f"Unknown metric '{cfg.metric}'. Valid: {list(_METRIC_NAMES)}")
    c.metric.type = metric_id

    # storage
    if isinstance(cfg.storage, LfsStorageConfig):
        c.storage.type = STORAGE_LFS
        s = c.storage.union.lfs
        s.device_path             = cfg.storage.device_path.encode()
        s.mount_point             = cfg.storage.mount_point.encode()
        s.lane_cache_size_kb      = cfg.storage.lane_cache_size_kb
        s.lane_size_mb            = cfg.storage.lane_size_mb
        s.log_size_mb             = cfg.storage.log_size_mb
        s.log_index_buckets       = cfg.storage.log_index_buckets
        s.object_cache_size_bytes = cfg.storage.object_cache_size_bytes
        s.read_ahead_size_bytes   = cfg.storage.read_ahead_size_bytes
        s.block_size              = cfg.storage.block_size
        s.flags                   = 0x0001 if cfg.storage.disable_crc else 0
    elif isinstance(cfg.storage, MemoryStorageConfig):
        c.storage.type = STORAGE_MEMORY
        m = c.storage.union.memory
        m.segments     = cfg.storage.segments
        m.segment_size = cfg.storage.segment_size
    else:
        raise ValueError(f"Unsupported storage type: {type(cfg.storage)}")

    # cache
    c.storage.cache.enabled               = 1 if cfg.cache.enabled else 0
    c.storage.cache.max_entries           = cfg.cache.max_entries
    c.storage.cache.max_object_size_bytes = cfg.cache.max_object_size_bytes

    # index
    c.index.type = INDEX_HNSW
    h = c.index.union.hnsw
    h.search_threads           = cfg.index.search_threads
    h.max_levels               = cfg.index.max_levels
    h.diversity_alpha          = cfg.index.diversity_alpha
    h.search_seen_set_capacity = cfg.index.search_seen_set_capacity
    h.ef_search                = cfg.index.ef_search
    for i, lv in enumerate(cfg.index.levels[:10]):
        h.level_config[i].M               = lv.M
        h.level_config[i].ef_construction = lv.ef_construction

    return c


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SearchResult:
    id: int
    distance: float


@dataclasses.dataclass
class DatabaseInfo:
    path: str
    version: Tuple[int, int, int]
    index_type: int
    metric_type: int
    vector_type: int


# ---------------------------------------------------------------------------
# Main NN20Db class
# ---------------------------------------------------------------------------

class NN20Db:
    """
    High-level handle to an nn20db database.

    Prefer using as a context manager::

        with NN20Db.open("/path/to/db") as db:
            ...
    """

    def __init__(self, ptr: int, dimension: int = 0, metadata_size: int = 0):
        self._ptr = ptr
        self._dimension   = dimension
        self._metadata_size = metadata_size

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, config: DatabaseConfig) -> "NN20Db":
        """Create a new database. Fails if the database already exists."""
        lib = _get_lib()
        c_cfg = _to_c_config(config)
        ptr   = ctypes.c_void_p(None)
        _check(lib.nn20db_create(ctypes.byref(c_cfg), ctypes.byref(ptr)), "create")
        return cls(ptr.value, config.dimension, config.metadata_size)

    @classmethod
    def open(cls, path: str) -> "NN20Db":
        """Open an existing database by path (uses saved config)."""
        lib = _get_lib()
        ptr = ctypes.c_void_p(None)
        _check(lib.nn20db_open(path.encode(), ctypes.byref(ptr)), f"open({path})")
        return cls(ptr.value)

    @classmethod
    def open_with_config(cls, config: DatabaseConfig) -> "NN20Db":
        """Open an existing database, overriding runtime parameters with config."""
        lib = _get_lib()
        c_cfg = _to_c_config(config)
        ptr   = ctypes.c_void_p(None)
        _check(lib.nn20db_open_with_config(ctypes.byref(c_cfg), ctypes.byref(ptr)), "open_with_config")
        return cls(ptr.value, config.dimension, config.metadata_size)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "NN20Db":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_ptr", None):
            try:
                _get_lib().nn20db_dtor(self._ptr)
            except Exception:
                pass
            self._ptr = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the database, freeing resources."""
        if self._ptr:
            _check(_get_lib().nn20db_close(self._ptr), "close")
            self._ptr = None

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def add(
        self,
        vector: Union[List[float], "bytes"],
        metadata: Optional[bytes] = None,
    ) -> None:
        """
        Add a vector to the database.

        Parameters
        ----------
        vector:
            List of floats or raw bytes (for bit vectors).
        metadata:
            Raw bytes to store alongside the vector (e.g. struct.pack('<i', idx)).
        """
        vec_buf  = _to_float_buf(vector) if not isinstance(vector, (bytes, bytearray)) else vector
        meta_ptr = ctypes.cast(metadata, ctypes.c_void_p) if metadata else None
        _check(_get_lib().nn20db_vector_add(self._ptr, vec_buf, meta_ptr), "vector_add")

    def search(
        self,
        query: Union[List[float], "bytes"],
        k: int,
    ) -> List[SearchResult]:
        """Search using the index's default ef_search."""
        return self._do_search(query, k, ef_search=None)

    def search_ef(
        self,
        query: Union[List[float], "bytes"],
        k: int,
        ef_search: int,
    ) -> List[SearchResult]:
        """Search with an explicit ef_search override."""
        return self._do_search(query, k, ef_search=ef_search)

    def _do_search(self, query, k, ef_search):
        lib      = _get_lib()
        q_buf    = _to_float_buf(query) if not isinstance(query, (bytes, bytearray)) else query
        results  = (_SearchResult * k)()
        if ef_search is None:
            _check(lib.nn20db_vector_search(self._ptr, q_buf, k, results), "vector_search")
        else:
            _check(lib.nn20db_vector_search_ef(self._ptr, q_buf, k, ef_search, results), "vector_search_ef")
        return [SearchResult(id=r.id, distance=r.distance) for r in results]

    def get(
        self,
        vector_id: int,
        dimension: Optional[int] = None,
        metadata_size: Optional[int] = None,
    ) -> Tuple[Optional[List[float]], Optional[bytes]]:
        """
        Retrieve a stored vector and its metadata by ID.

        Returns (vector_floats, metadata_bytes). Pass dimension / metadata_size
        if not set on the NN20Db instance.
        """
        dim  = dimension    or self._dimension
        msz  = metadata_size or self._metadata_size
        v_buf = (ctypes.c_float * dim)() if dim  else None
        m_buf = (ctypes.c_uint8 * msz)() if msz else None
        _check(
            _get_lib().nn20db_vector_get(
                self._ptr, ctypes.c_uint64(vector_id),
                v_buf, m_buf,
            ),
            f"vector_get({vector_id})",
        )
        vec  = list(v_buf) if v_buf else None
        meta = bytes(m_buf) if m_buf else None
        return vec, meta

    def remove(self, vector_id: int) -> None:
        """Remove a vector by ID."""
        _check(_get_lib().nn20db_vector_remove(self._ptr, ctypes.c_uint64(vector_id)), f"vector_remove({vector_id})")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Flush pending writes to storage."""
        _check(_get_lib().nn20db_sync(self._ptr), "sync")

    def compact(self) -> None:
        """Run a compaction pass to reclaim space and merge log entries."""
        _check(_get_lib().nn20db_compact(self._ptr), "compact")


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def list_databases(path: str, mount_point: str = "") -> List[DatabaseInfo]:
    """
    Recursively find all nn20db databases under *path*.

    Parameters
    ----------
    path:
        Root directory to search.
    mount_point:
        VFS mount point (ESP32 only). Pass "" on Linux.
    """
    lib = _get_lib()
    out = ctypes.POINTER(_DbListResult)(None)
    n   = lib.nn20db_list_databases(
        path.encode(), mount_point.encode() if mount_point else b"",
        ctypes.byref(out),
    )
    if n < 0:
        raise Nn20dbError(n, "list_databases")
    results = []
    for i in range(n):
        r = out[i]
        results.append(DatabaseInfo(
            path        = r.path.decode(),
            version     = (r.version_major, r.version_minor, r.version_patch),
            index_type  = r.index_type,
            metric_type = r.metric_type,
            vector_type = r.vector_type,
        ))
    if n > 0 and out:
        ctypes.cdll.LoadLibrary("libc.so.6").free(out)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float_buf(vec) -> ctypes.Array:
    """Convert list/tuple/numpy array to a ctypes c_float array."""
    try:
        import numpy as np
        if isinstance(vec, np.ndarray):
            arr = vec.astype(np.float32)
            return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    except ImportError:
        pass
    floats = (ctypes.c_float * len(vec))(*vec)
    return floats
