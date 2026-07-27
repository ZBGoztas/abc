"""
NPZ Store (BRepNet tensor dosyalari)
--------------------------------------
config.py'deki npz_schema_config'e baglidir.
Gorevi: her parca icin tensor tensorlarini {part_id}.npz olarak
kaydetmek / okumak, schema_version disiplinini korumak.
"""
from typing import Dict
import numpy as np

from src.config import npz_schema_config
from src.braketLog.logger import get_logger
logger = get_logger(__name__)


class NpzStore:

    def __init__(self, config=None):
        self.config = config or npz_schema_config
        logger.debug(f"NpzStore baslatildi. NPZ_DIR = {self.config.NPZ_DIR}")

    def save(self, part_id: str, tensors: Dict[str, np.ndarray]) -> None:
        """
        tensors icine schema_version yoksa otomatik ekler.
        Zorunlu anahtarlar eksikse hata verir.
        """
        tensors = dict(tensors)
        tensors.setdefault(
            "schema_version",\
            np.array([self.config.SCHEMA_VERSION], dtype="int16"),
        )

        try: 
            self.config.validate_keys(tensors)
        except ValueError as e:
            logger.error(f"NPZ schema contract ihlali (save): part_id{part_id} | {e}")
            raise
        path = self.config.path_for(part_id)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **tensors)
            logger.info(f"NPZ kaydedildi: part_id={part_id} -> {path}")
        except Exception as e:
            logger.critical(f"NPZ kaydetme HATASI: part_id={part_id} | {e}")
            raise

    def load(self, part_id: str) -> Dict[str, np.ndarray]:
        path = self.config.path_for(part_id)
        if not path.exists():
            logger.error(f"NPZ dosyasi bulunamadi: {path}")
            raise FileNotFoundError(f"NPZ dosyasi bulunamadi: {path}")

        try:
            with np.load(path) as data:
                # validate_keys ve dict'e kopyalama islemleri
                # 'data' hala acikken (with blogu icinde) yapiliyor.
                self.config.validate_keys(data)
                loaded_tensors = {k: data[k] for k in data.files}
                logger.debug(f"NPZ yuklendi: part_id={part_id}, anahtarlar={data.files}")
        except ValueError as e:
            logger.error(f"NPZ schema contract ihlali (load): part_id={part_id} | {e}")
            raise
        except Exception as e:
            logger.critical(f"NPZ okuma HATASI: part_id={part_id} | {e}")
            raise

        return loaded_tensors
    def exists(self, part_id: str) -> bool:
        found = self.config.path_for(part_id).exists()
        logger.debug(f"exists kontrolu: part_id={part_id} -> {found}")
        return found

    def delete(self, part_id: str) -> None:
        path = self.config.path_for(part_id)
        if path.exists():
            path.unlink()
            logger.info(f"NPZ silindi: part_id={part_id}")
        else:
            logger.warning(f"Silinecek NPZ bulunamadi: part_id={part_id}")