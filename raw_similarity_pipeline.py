"""
raw_similarity_pipeline.py
--------------------------
1) data/abc/raw klasorundeki STEP dosyalarini normalize eder.
2) Her normalize edilen dosya icin FAISS index uzerinde top_k benzer parcayi bulur.
3) Bulunan dataset dosyalarini data/abc/raw_similar/<query_part>/ altina kopyalar.

Kullanim:
    python raw_similarity_pipeline.py --top_k 5
"""

import argparse
import csv
import shutil
from pathlib import Path

from src.config import BASE_DIR
from src.embedding_store import EmbeddingStore
from src.metadata_store import MetadataStore
from src.braketLog.logger import get_logger
from find_similar_parts import normalize_query_step, generate_query_embedding

logger = get_logger(__name__)

ABC_DATA_DIR = BASE_DIR / "data" / "abc"
RAW_DIR = ABC_DATA_DIR / "raw"
RAW_NORMALIZED_DIR = ABC_DATA_DIR / "processed" / "raw_normalized"
RAW_SIMILAR_DIR = ABC_DATA_DIR / "raw_similar"
RAW_SIMILAR_MANIFEST = RAW_SIMILAR_DIR / "manifest.csv"
STEP_SUFFIXES = {".step", ".stp"}


def list_step_files(folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in STEP_SUFFIXES
        ]
    )


def normalize_raw_files(raw_dir: Path, normalized_dir: Path) -> list[Path]:
    raw_files = list_step_files(raw_dir)
    if not raw_files:
        logger.warning(f"Raw klasorunde STEP dosyasi bulunamadi: {raw_dir}")
        return []

    normalized_dir.mkdir(parents=True, exist_ok=True)

    normalized_files: list[Path] = []
    for step_path in raw_files:
        logger.info(f"Normalize ediliyor: {step_path.name}")
        normalized_path = normalize_query_step(step_path, output_dir=normalized_dir)
        normalized_files.append(normalized_path)

    return normalized_files


def search_similar_parts(
    normalized_step_path: Path,
    top_k: int,
    embedding_store: EmbeddingStore,
    metadata_df,
) -> list[dict]:
    query_embedding, _ = generate_query_embedding(normalized_step_path)
    search_results = embedding_store.search(query_embedding, top_k=top_k)

    if search_results.empty:
        logger.warning(f"Benzer parca bulunamadi: {normalized_step_path.name}")
        return []

    enriched_rows: list[dict] = []
    for _, row in search_results.iterrows():
        part_id = row["part_id"]
        distance = float(row["distance"])
        meta_row = metadata_df[metadata_df["part_id"] == part_id]

        info = {
            "part_id": part_id,
            "distance": distance,
            "source_path": None,
        }
        if not meta_row.empty:
            info["source_path"] = meta_row.iloc[0].get("source_path")
        enriched_rows.append(info)

    return enriched_rows


def copy_similar_files(
    query_path: Path,
    results: list[dict],
    output_root: Path,
) -> list[dict]:
    query_dir = output_root / query_path.stem
    query_dir.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict] = []
    for rank, info in enumerate(results, start=1):
        source_path = info.get("source_path")
        if not source_path:
            logger.warning(
                f"{query_path.name}: part_id={info.get('part_id')} icin source_path bulunamadi"
            )
            continue

        source_file = Path(source_path)
        if not source_file.exists():
            logger.warning(
                f"{query_path.name}: source dosya bulunamadi -> {source_file}"
            )
            continue

        target_path = query_dir / f"{rank:02d}__{source_file.name}"
        shutil.copy2(source_file, target_path)

        copied_rows.append(
            {
                "query_part_id": query_path.stem,
                "query_file": query_path.name,
                "rank": rank,
                "matched_part_id": info.get("part_id"),
                "distance": info.get("distance"),
                "source_path": str(source_file),
                "copied_path": str(target_path),
            }
        )

        logger.info(
            f"{query_path.name}: benzer dosya kopyalandi -> {target_path.name} "
            f"(rank={rank}, distance={info.get('distance'):.4f})"
        )

    return copied_rows


def write_manifest(rows: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_part_id",
        "query_file",
        "rank",
        "matched_part_id",
        "distance",
        "source_path",
        "copied_path",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    normalized_dir: Path = RAW_NORMALIZED_DIR,
    output_dir: Path = RAW_SIMILAR_DIR,
    top_k: int = 5,
) -> None:
    normalized_files = normalize_raw_files(raw_dir, normalized_dir)
    if not normalized_files:
        return

    embedding_store = EmbeddingStore()
    metadata_store = MetadataStore()
    metadata_df = metadata_store.read_all()

    if metadata_df.empty:
        logger.warning(
            "Metadata bos. Benzerlik eslesmelerinin source_path bilgisi eksik kalabilir."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for normalized_path in normalized_files:
        logger.info(f"Benzer parcalar aranıyor: {normalized_path.name}")
        results = search_similar_parts(
            normalized_path,
            top_k=top_k,
            embedding_store=embedding_store,
            metadata_df=metadata_df,
        )
        manifest_rows.extend(copy_similar_files(normalized_path, results, output_dir))

    write_manifest(manifest_rows, RAW_SIMILAR_MANIFEST)
    logger.info(
        f"Pipeline tamamlandi | normalize_edilen={len(normalized_files)} | "
        f"kopyalanan_dosya={len(manifest_rows)} | manifest={RAW_SIMILAR_MANIFEST}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw klasorunu normalize eder ve dataset icinden benzer dosyalari ayirir"
    )
    parser.add_argument("--top_k", type=int, default=5, help="Her raw dosya icin kac benzer dosya kopyalansin")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DIR), help="Normalize edilecek raw klasoru")
    parser.add_argument(
        "--normalized_dir",
        type=str,
        default=str(RAW_NORMALIZED_DIR),
        help="Normalize edilmis raw dosyalarinin kaydedilecegi klasor",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(RAW_SIMILAR_DIR),
        help="Benzer dataset dosyalarinin kopyalanacagi klasor",
    )
    args = parser.parse_args()

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        normalized_dir=Path(args.normalized_dir),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
