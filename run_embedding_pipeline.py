"""
run_embedding_pipeline.py
--------------------------
Asama 1: STEP (orientation+scale normalized) -> FeatureEngine -> tensor_builder
        -> NpzStore -> embedding_generator -> EmbeddingStore -> MetadataStore
"""

from pathlib import Path
import datetime as dt

from src.config import BASE_DIR
from src.service.feature_engine import FeatureEngine
from src.service.tensor_builder import build_tensors
from src.npz_store import NpzStore
from src.service.embedding_generator import generate_embedding
from src.embedding_store import EmbeddingStore
from src.metadata_store import MetadataStore, compute_sha256, now_iso
from src.braketLog.logger import get_logger

logger = get_logger(__name__)

# Adim 1+2'nin ciktisi olan klasor (main.py'deki ile ayni)
NORMALIZED_STEP_DIR = BASE_DIR / "data" / "oryantasyon_and_scale_normalized_step"

npz_store = NpzStore()
embedding_store = EmbeddingStore()
metadata_store = MetadataStore()


def process_one(step_path: Path) -> bool:
    part_id = step_path.stem  # dosya adi (uzantisiz) -> part_id

    try:
        # --- 1) METADATA: daha once islenmis mi kontrol et (idempotency) ---
        sha256 = compute_sha256(str(step_path))
        if metadata_store.exists_by_sha256(sha256):
            logger.info(f"{part_id}: sha256 zaten indexli, atlaniyor")
            return True

        # --- 2) FEATURE ENGINE ---
        fe = FeatureEngine(str(step_path))

        # --- 3) TENSOR BUILD ---
        tensors = build_tensors(fe)

        # --- 4) NPZ KAYDET ---
        npz_store.save(part_id, tensors)

        # --- 5) EMBEDDING URET + FAISS'E EKLE ---
        if not embedding_store.part_exists(part_id):
            embedding = generate_embedding(tensors, part_id=part_id)
            embedding_store.add(part_id, embedding)
        else:
            logger.warning(f"{part_id}: embedding zaten mevcut, tekrar eklenmedi")

        # --- 6) METADATA KAYDET ---
        bbox = fe.bounding_box()
        fiziksel = fe.physical_features()
        yuzey_tipleri = fe.yuzey_tipleri_tespit_et()
        topoloji = fe.topoloji_gez()

        row = {
            "part_id": part_id,
            "file_name": step_path.name,
            "source_path": str(step_path),
            "bbox_x": bbox["x"], "bbox_y": bbox["y"], "bbox_z": bbox["z"],
            "center_x": bbox["merkez"][0], "center_y": bbox["merkez"][1], "center_z": bbox["merkez"][2],
            "volume": fiziksel["hacim"],
            "face_count": len(topoloji["faces"]),
            "edge_count": len(topoloji["edges"]),
            "node_count": len(topoloji["nodes"]),
            "plane_face_count": len(yuzey_tipleri["duzYuzeyler"]),
            "cylinder_face_count": len(yuzey_tipleri["silindirikYuzeyler"]),
            "bspline_face_count": len(yuzey_tipleri["BSplineYuzeyler"]),
            "feature_version": 1,
            "embedding_version": embedding_store.config.EMBEDDING_VERSION,
            "created_at": now_iso(),
            "source_sha256": sha256,
        }
        metadata_store.upsert(row)

        logger.info(f"{part_id}: pipeline tamamlandi (NPZ + embedding + metadata)")
        return True

    except Exception:
        logger.exception(f"{part_id}: pipeline hata ile sonuclandi")
        return False


def main():
    step_files = sorted(
        list(NORMALIZED_STEP_DIR.glob("*.step")) + list(NORMALIZED_STEP_DIR.glob("*.stp"))
    )
    if not step_files:
        logger.error(f"Normalize edilmis STEP dosyasi bulunamadi: {NORMALIZED_STEP_DIR}")
        return

    basarili, basarisiz = 0, 0
    for step_path in step_files:
        logger.info(f"Isleniyor: {step_path.name}")
        if process_one(step_path):
            basarili += 1
        else:
            basarisiz += 1

    # --- HEPSI BITTIKTEN SONRA INDEX'I DISKE YAZ ---
    embedding_store.save_index()
    logger.info(f"Pipeline tamamlandi | Basarili: {basarili} | Basarisiz: {basarisiz}")


if __name__ == "__main__":
    main()
