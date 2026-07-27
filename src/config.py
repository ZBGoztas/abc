import logging
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
Step_Data_Path = BASE_DIR / "stepData"
############ LOG AYARLARI #########################
@dataclass
class LogConfig:
    LOG_DIR      : Path  = BASE_DIR / 'logs'
    LOG_LVL      : int  = logging.DEBUG
    LOG_FORMAT   : str  = "%(asctime)s | %(filename)-25s | %(funcName)-25s | LINE:%(lineno)-4d | %(levelname)-8s | %(message)s"
    DATE_FORMAT  : str  = "%Y-%m-%d %H:%M"
    LOG_ROTATING : bool = False
    MAX_BYTES    : int  = 5 * 1024 * 1024   # 5 MB
    BACKUP_COUNT : int  = 3

    # ✅ Key'ler artık integer (logging sabitleri)
    LOG_FILE: dict = field(default_factory=lambda: {
        logging.DEBUG    : 'debug.log',
        logging.INFO     : 'info.log',
        logging.WARNING  : 'warning.log',
        logging.ERROR    : 'error.log',
        logging.CRITICAL : 'critical.log',
    })

# ✅ Tek instance — her yerden import edilir
log_config = LogConfig()
############ LOG AYARLARI #########################


############ METADATA SCHEMA AYARLARI ##############
@dataclass
class MetadataSchemaConfig:
    """
    Parquet metadata store icin schema contract.
    schema_version degisirse -> yeni bir config instance'i (v2) olustur,
    eski kod/pipeline'i bozma.
    """

    SCHEMA_VERSION: int = 1

    PARQUET_PATH: Path = BASE_DIR /"data" / "metadata" / "metadata.parquet"

    # kolon -> dtype eslemesi 
    DTYPES: dict = field(default_factory=lambda: {
        "part_id"             : "string",
        "file_name"           : "string",
        "source_path"         : "string",
        "bbox_x"              : "float64",
        "bbox_y"              : "float64",
        "bbox_z"              : "float64",
        "center_x"            : "float64",
        "center_y"            : "float64",
        "center_z"            : "float64",
        "volume"              : "float64",
        "face_count"          : "int32",
        "edge_count"          : "int32",
        "node_count"          : "int32",
        "plane_face_count"    : "int32",
        "cylinder_face_count" : "int32",
        "bspline_face_count"  : "int32",
        "feature_version"     : "int32",
        "embedding_version"   : "int32",
        "created_at"          : "string",
        "source_sha256"       : "string",
        "schema_version"      : "int32",
    })

    # Her satirda zorunlu olan kolanlar
    REQUIRED_COLUMNS: list = field(default_factory= lambda:[
        "part_id",
        "file_name",
        "source_path",
        "bbox_x", "bbox_y", "bbox_z",
        "center_x", "center_y", "center_z",
        "volume",
        "face_count", "edge_count", "node_count",
        "plane_face_count", "cylinder_face_count", "bspline_face_count",
        "feature_version",
        "embedding_version",
        "created_at",
        "source_sha256",
    ])

    # Schema contract'in cekirdek 4 alani (rehberdeki zorunlu kurallar)
    CRITICAL_COLUMNS: list = field(default_factory=lambda: [
        "part_id",
        "feature_version",
        "schema_version",
        "source_sha256",
    ])

    def validate_row(self, row: dict) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"Zorunlu kolonlar :{missing}")
        
        for c in self.CRITICAL_COLUMNS:
            if c == "schema_version":
                continue # otomatik eklenir, kontrol gerektirmez
            if row.get(c) in (None, ""):
                raise ValueError(f"'{c}' bos olamaz (schema contract ihlali)")
    def __post_init__(self):
        self.PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

metadata_schema_config = MetadataSchemaConfig()
############ METADATA SCHEMA AYARLARI ##############



@dataclass
class NpzSchemaConfig:
    """
    BRepNet tensorlarini icin NPZ dosya schema contract'i.
    """
    SCHEMA_VERSION: int = 1

    NPZ_DIR: Path = BASE_DIR / "data" / "features_npz"

    # anahtar -> (dtype, shape aciklamasi)
    KEYS: dict = field(default_factory=lambda: {
        "face_features"        : "float32",   # [F, Df]
        "edge_features"        : "float32",   # [E, De]
        "edge_to_face"         : "int32",      # [E, 2]
        "face_adjacency_index" : "int32",      # [2, K]
        "face_types"           : "int16",      # [F]
        "normals"              : "float32",   # [F, G, 3]
        "normals_mask"         : "bool",       # [F, G]
        "bbox"                 : "float32",    # [3]
        "center"               : "float32",    # [3]
        "volume"               : "float64",    # [1]
        "schema_version"       : "int16",      # [1]
    })

    REQUIRED_KEYS: list = field(default_factory=lambda: [
        "face_features",
        "edge_features",
        "edge_to_face",
        "face_adjacency_index",
        "face_types",
        "bbox",
        "center",
        "volume",
        "schema_version",
    ])


    def validate_keys(self, npz_data) -> None:
        missing = [k for k in self.REQUIRED_KEYS if k not in npz_data]
        if missing:
            raise ValueError(f"NPZ'de zorunlu anahtarlar eksik: {missing}")
        
    def __post_init__(self):
        self.NPZ_DIR.mkdir(parents=True, exist_ok=True)
    def path_for(self, part_id: str) -> Path:
        return self.NPZ_DIR / f"{part_id}.npz"
        

npz_schema_config = NpzSchemaConfig()
############ NPZ SCHEMA AYARLARI ####################


############ EMBEDDING / FAISS SCHEMA AYARLARI ######
@dataclass
class EmbeddingConfig:
    EMBEDDING_VERSION: int = 1
    
    FAISS_INDEX_PATH: Path = BASE_DIR / "index" / "faiss.index"
    ID_MAP_PATH: Path = BASE_DIR / "index" / "id_map.parquet"
    EMBEDDING_SNAPSHOT_PATH: Path = BASE_DIR / "data" / "embeddings" / "embedding_snapshot.parquet"

    EMBEDDING_DIM: int = 256 # vektor uzunlugu
    def __post_init__(self):
        self.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.EMBEDDING_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True) 

embedding_config = EmbeddingConfig()
############ EMBEDDING / FAISS SCHEMA AYARLARI ######
