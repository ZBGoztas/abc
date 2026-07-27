"""
find_similar_parts.py
-----------------------
Kullanici bir STEP dosyasi verir -> pipeline'dan gecirilir -> 
FAISS'te en benzer parcalar bulunur -> sonuclar goruntulenir.

Kullanim:
    python find_similar_parts.py path/to/yeni_parca.stp --top_k 5 --show
"""

import argparse
import shutil
from pathlib import Path

from src.config import BASE_DIR
from src.service.orientation_normalizasyon import atalet_eksenlerine_gore_normalizasyon
from src.service.scale_normalize import olcek_normalizasyonu
from src.service.step_file_processes import stepReader, shape_to_step
from src.service.feature_engine import FeatureEngine
from src.service.tensor_builder import build_tensors
from src.service.embedding_generator import generate_embedding
from src.embedding_store import EmbeddingStore
from src.metadata_store import MetadataStore
from src.braketLog.logger import get_logger

logger = get_logger(__name__)

TEMP_DIR = BASE_DIR / "data" / "_temp_query"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def normalize_query_step(step_path: Path, output_dir: Path = TEMP_DIR) -> Path:
    """
    Girdi STEP dosyasini egitim setiyle AYNI normalize zincirinden gecirir.
    Cikti olarak normalize edilmis STEP dosyasi doner.
    """
    file_name = step_path.name

    shape = stepReader(str(step_path))
    if shape is None:
        raise ValueError(f"{file_name}: Shape okunamadi")

    oryantasyon_shape = atalet_eksenlerine_gore_normalizasyon(file_name, shape)
    if oryantasyon_shape is None or oryantasyon_shape.IsNull():
        raise ValueError(f"{file_name}: Orientation normalize basarisiz")

    scale_shape = olcek_normalizasyonu(file_name, oryantasyon_shape)
    if scale_shape is None or scale_shape.IsNull():
        raise ValueError(f"{file_name}: Scale normalize basarisiz")

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / file_name
    success = shape_to_step(file_name, scale_shape, str(normalized_path))
    if not success:
        raise ValueError(f"{file_name}: Normalize edilmis STEP kaydedilemedi")

    return normalized_path


def generate_query_embedding(normalized_step_path: Path):
    """
    Normalize edilmis STEP dosyasindan embedding uretir (diske NPZ yazmadan).
    """
    part_id = normalized_step_path.stem
    fe = FeatureEngine(str(normalized_step_path))
    tensors = build_tensors(fe)
    embedding = generate_embedding(tensors, part_id=part_id)
    return embedding, fe


def find_similar(step_path: str, top_k: int = 5):
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {step_path}")

    logger.info(f"Sorgu dosyasi normalize ediliyor: {step_path.name}")
    normalized_path = normalize_query_step(step_path)

    logger.info(f"Embedding uretiliyor: {step_path.name}")
    query_embedding, fe = generate_query_embedding(normalized_path)

    embedding_store = EmbeddingStore()
    results = embedding_store.search(query_embedding, top_k=top_k)

    metadata_store = MetadataStore()
    meta_df = metadata_store.read_all()

    print(f"\n{'='*70}")
    print(f"SORGU: {step_path.name}")
    print(f"{'='*70}")

    enriched_rows = []
    for _, row in results.iterrows():
        part_id = row["part_id"]
        distance = row["distance"]
        meta_row = meta_df[meta_df["part_id"] == part_id]

        info = {
            "part_id": part_id,
            "distance": round(float(distance), 4),
        }
        if not meta_row.empty:
            info["volume"] = round(float(meta_row.iloc[0]["volume"]), 6)
            info["face_count"] = int(meta_row.iloc[0]["face_count"])
            info["source_path"] = meta_row.iloc[0]["source_path"]
        enriched_rows.append(info)

    for i, info in enumerate(enriched_rows, start=1):
        print(f"{i}. part_id={info['part_id']:<15} distance={info['distance']:<8} "
              f"face_count={info.get('face_count','?'):<6} volume={info.get('volume','?')}")

    # temizlik
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    return enriched_rows


def show_step_files(results, top_n: int = 3):
    """
    En benzer ilk N parcanin STEP dosyalarini pythonOCC penceresinde acar (opsiyonel).
    """
    try:
        from OCC.Display.SimpleGui import init_display
    except ImportError:
        logger.warning("pythonOCC display modulu bulunamadi, gorsellestirme atlandi")
        return

    for info in results[:top_n]:
        source_path = info.get("source_path")
        if not source_path or not Path(source_path).exists():
            continue

        shape = stepReader(source_path)
        if shape is None:
            continue

        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(shape, update=True)
        display.FitAll()
        print(f"Goruntuleniyor: {info['part_id']} (kapatmak icin pencereyi kapatin)")
        start_display()


def main():
    parser = argparse.ArgumentParser(description="Bir STEP dosyasina en benzer parcalari bulur")
    parser.add_argument("step_path", type=str, help="Sorgulanacak STEP dosyasinin yolu")
    parser.add_argument("--top_k", type=int, default=5, help="Kac benzer parca gosterilsin")
    parser.add_argument("--show", action="store_true", help="Sonuclari 3B pencerede goruntule")
    args = parser.parse_args()

    results = find_similar(args.step_path, top_k=args.top_k)

    if args.show:
        show_step_files(results, top_n=3)


if __name__ == "__main__":
    main()
