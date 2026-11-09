from pathlib import Path
import py7zr
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = BASE_DIR / "data" / "abc" / "raw" / "step" / "retrieve abc_0001_step_v00.7z"
TEMP_EXTRACT_DIR = BASE_DIR / "data" / "abc" / "raw" / "extracted"
OUTPUT_DIR = BASE_DIR / "data" / "abc" / "raw" / "stepfiles"

def extract_and_export(archive_path: Path) -> None:
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=TEMP_EXTRACT_DIR)

    step_files = sorted(
        p for p in TEMP_EXTRACT_DIR.rglob("*")
        if p.suffix.lower() in {".step", ".stp"}
    )

    for idx, src in enumerate(step_files, start=1):
        dest = OUTPUT_DIR / f"step_{idx}.stp"
        shutil.copy2(src, dest)
        print(f"Exported: {dest}")


if __name__ == "__main__":
    extract_and_export(ARCHIVE_PATH)