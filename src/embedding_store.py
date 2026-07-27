"""
Embedding Store (FAISS + Parquet yedek)
------------------------------------------
config.py'deki embedding_config'e baglidir.
Gorevi:
- Embedding vektorlerini FAISS indexe eklemek
- faiss_id <-> part_id eslesmesini id_map.parquet'e yazmak
- Denetlenebilirlik icin embedding_snapshot.parquet tutmak

Not: faiss kutuphanesi sisteminde kurulu olmali (pip install faiss-cpu).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None

from src.config import embedding_config
from src.braketLog.logger import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    def __init__(self, config=None):
        self.config = config or embedding_config

        if faiss is None:
            logger.critical("faiss kurulu degil. 'pip install faiss-cpu' ile kurulum yapin.")
            raise ImportError(
                "faiss kurulu degil. 'pip install faiss-cpu' ile kurulum yapin."
            )

        self.index = self._load_or_create_index()
        self.id_map = self._load_id_map()
        logger.debug(
            f"EmbeddingStore baslatildi. ntotal={self.index.ntotal}, "
            f"id_map_satir={len(self.id_map)}"
        )

    # ---------- INDEX YONETIMI ----------

    def _load_or_create_index(self):
        if self.config.FAISS_INDEX_PATH.exists():
            logger.debug(f"Mevcut FAISS index yukleniyor: {self.config.FAISS_INDEX_PATH}")
            return faiss.read_index(str(self.config.FAISS_INDEX_PATH))
        logger.warning(
            f"FAISS index bulunamadi, yeni bos index olusturuluyor "
            f"(dim={self.config.EMBEDDING_DIM})."
        )
        return faiss.IndexFlatL2(self.config.EMBEDDING_DIM)

    def _load_id_map(self) -> pd.DataFrame:
        if self.config.ID_MAP_PATH.exists():
            df = pd.read_parquet(self.config.ID_MAP_PATH)
            logger.debug(f"id_map yuklendi: {len(df)} satir.")
            return df
        logger.warning(f"id_map parquet bulunamadi, bos DataFrame olusturuluyor: {self.config.ID_MAP_PATH}")
        return pd.DataFrame(columns=["faiss_id", "part_id", "embedding_version"])

    def save_index(self) -> None:
        try:
            self.config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.config.ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.config.FAISS_INDEX_PATH))
            self.id_map.to_parquet(self.config.ID_MAP_PATH, engine="pyarrow", index=False)
            logger.info(
                f"FAISS index ve id_map kaydedildi. ntotal={self.index.ntotal}, "
                f"id_map_satir={len(self.id_map)}"
            )
        except Exception as e:
            logger.critical(f"Index/id_map kaydetme HATASI: {e}")
            raise

    # ---------- EKLEME ----------

    def add(self, part_id: str, vector: np.ndarray) -> None:
        """
        Tek bir parca icin embedding ekler.
        vector: shape [EMBEDDING_DIM], dtype float32
        """
        vector = np.asarray(vector, dtype="float32").reshape(1, -1)
        if vector.shape[1] != self.config.EMBEDDING_DIM:
            logger.error(
                f"Embedding boyutu uyusmuyor: beklenen {self.config.EMBEDDING_DIM}, "
                f"gelen {vector.shape[1]} | part_id={part_id}"
            )
            raise ValueError(
                f"Embedding boyutu uyusmuyor: beklenen {self.config.EMBEDDING_DIM}, "
                f"gelen {vector.shape[1]}"
            )

        faiss_id = self.index.ntotal
        self.index.add(vector)

        new_row = pd.DataFrame([{
            "faiss_id": faiss_id,
            "part_id": part_id,
            "embedding_version": self.config.EMBEDDING_VERSION,
        }])
        self.id_map = pd.concat([self.id_map, new_row], ignore_index=True)

        logger.info(f"Embedding eklendi: part_id={part_id}, faiss_id={faiss_id}")

        self._append_snapshot(part_id, vector[0])

    def _append_snapshot(self, part_id: str, vector: np.ndarray) -> None:
        """Denetlenebilirlik icin embedding_snapshot.parquet'e yazar."""
        snapshot_path = self.config.EMBEDDING_SNAPSHOT_PATH

        if snapshot_path.exists():
            df = pd.read_parquet(snapshot_path)
        else:
            logger.warning(f"Snapshot parquet bulunamadi, yeni olusturuluyor: {snapshot_path}")
            df = pd.DataFrame(columns=["part_id", "embedding_version", "vector"])

        new_row = pd.DataFrame([{
            "part_id": part_id,
            "embedding_version": self.config.EMBEDDING_VERSION,
            "vector": vector.tolist(),
        }])
        df = pd.concat([df, new_row], ignore_index=True)

        try:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(snapshot_path, engine="pyarrow", index=False)
            logger.debug(f"Snapshot guncellendi: part_id={part_id} -> {snapshot_path}")
        except Exception as e:
            logger.critical(f"Snapshot yazma HATASI: part_id={part_id} | {e}")
            raise

    # ---------- ARAMA ----------

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """
        Top-k benzer parcayi getirir.
        Donen DataFrame: part_id, distance, faiss_id
        """
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            match = self.id_map[self.id_map["faiss_id"] == idx]
            part_id = match.iloc[0]["part_id"] if not match.empty else None
            results.append({"part_id": part_id, "distance": float(dist), "faiss_id": int(idx)})

        logger.debug(f"Arama tamamlandi: top_k={top_k}, bulunan={len(results)}")
        return pd.DataFrame(results)

    def part_exists(self, part_id: str) -> bool:
        found = bool((self.id_map["part_id"] == part_id).any())
        logger.debug(f"part_exists kontrolu: part_id={part_id} -> {found}")
        return found


