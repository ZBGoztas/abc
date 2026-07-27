"""
Metadata Store (Parquet-backed)
--------------------------------
config.py'deki metadata_schema_config'e baglidir.
Gorevi: parquet dosyasina satir eklemek / okumak / guncellemek.
Schema kurallari config'te, I/O mantigi burada.
"""

import hashlib
import datetime as dt
from typing import Optional

import pandas as pd

from src.config import metadata_schema_config
from src.braketLog.logger import get_logger

logger = get_logger(__name__)

def compute_sha256(file_path: str) -> str:
    """Kaynak STEP dosyasinin sha256'sini hesaplar."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return dt.datetime.utcnow().isoformat()

class MetadataStore:

    def __init__(self, config=None):
        self.config = config or metadata_schema_config
        self.parquet_path = self.config.PARQUET_PATH

    # -------------- READ ----------------------------

    def read_all(self) -> pd.DataFrame:

        if not self.parquet_path.exists():
            logger.warning(f"Parquet dosyasi henuz yok, bos DataFrame donuluyor: {self.parquet_path}")
            cols = self.config.REQUIRED_COLUMNS + ["schema_version"]
            return pd.DataFrame(columns=cols)
        
        df = pd.read_parquet(self.parquet_path)
        logger.debug(f"Parquet okundu, {len(df)} satir bulundu.")
        return df
    
    def read_by_part_id(self, part_id: str) -> Optional[pd.Series]:
        df = self.read_all()
        match = df[df["part_id"] == part_id]
        if match.empty:
            logger.debug(f"part_id bulunamadi: {part_id}")
            return None
        return match.iloc[0]
    

    def exists(self, part_id: str) -> bool:
        return self.read_by_part_id(part_id) is not None
    
    def exists_by_sha256(self, source_sha256: str) -> bool:
        """Ayni dosyanin tekrar indexlenmesini engellemek icin."""
        df = self.read_all()

        if df.empty or "source_sha256" not in df.columns:
            return False
        found = bool((df["source_sha256"] == source_sha256).any())
        if found:
            logger.info(f"source_sha256 zaten indexli: {source_sha256[:12]}...")
        return found
    
    # ----------- WRITE ---------------------

    def append(self, row: dict) -> None:
        """Yeni parca ekler. part_id zaten varsa hata verir."""

        row = self._with_schema_version(row)

        try:
            self.config.validate_row(row)
        except ValueError as e:
            logger.error(f"Schema contract ihlali (append): {e}")
            raise
        
        if self.exists(row["part_id"]):
            logger.error(f"part_id zaten mevcut, append basarisiz: {row['part_id']}")
            raise ValueError(
                f"part_id '{row['part_id']}' zaten mevcut. upsert() kullan."
            )
        
        df = self.read_all()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write(df)
        logger.info(f"Yeni satir eklendi: part_id={row['part_id']}")

    def upsert(self, row: dict) -> None:
        """part_id varsa satiri gunceller, yoksa ekler."""

        row = self._with_schema_version(row)

        try:
            self.config.validate_row(row)
        except ValueError as e:
            logger.error(f"Schema contract ihlali (upsert): {e}")
            raise

        df = self.read_all()
        if df.empty:
            df = pd.DataFrame([row])
            logger.info(f"Ilk satir olusturuldu: part_id={row['part_id']}")
        else:
            mask = df["part_id"] == row["part_id"]
            if mask.any():
                for k, v in row.items():
                    df.loc[mask, k] = v
                logger.info(f"Satir guncellendi: part_id={row['part_id']}")
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                logger.info(f"Yeni satir eklendi (upsert): part_id={row['part_id']}")

        self._write(df)

    def delete(self, part_id: str) -> None:
        df = self.read_all()
        before = len(df)
        df = df[df["part_id"] != part_id]
        after = len(df)

        if before == after:
            logger.warning(f"Silinecek part_id bulunamadi: {part_id}")
        else:
            logger.info(f"Satir silindi: part_id={part_id}")

        self._write(df)

    # ---------- INTERNAL ----------

    def _with_schema_version(self, row: dict) -> dict:
        row = dict(row)
        row["schema_version"] = self.config.SCHEMA_VERSION
        return row
    
    def _write(self, df: pd.DataFrame) -> None:
        for col, dtype in self.config.DTYPES.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Dtype donusumu basarisiz, kolon atlandi: {col} ({e})")
        
        try:
            self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(self.parquet_path, engine="pyarrow", index=False)
            logger.debug(f"Parquet yazildi: {self.parquet_path} ({len(df)} satir)")
        except Exception as e:
            logger.critical(f"Parquet yazma HATASI: {e}")
            raise
