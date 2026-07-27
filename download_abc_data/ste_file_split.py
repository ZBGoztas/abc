import hashlib
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_SOLID

INPUT_DIR = Path(r"<local-path>")
OUTPUT_DIR = Path(r"<local-path>")
MAX_FILES = None
HASH_LOG = OUTPUT_DIR / "processed_hashes.txt"


def get_next_step_index(out_dir: Path) -> int:
    """Return next global index for files named as step_<n>.stp in OUTPUT_DIR."""
    max_index = 0
    for p in out_dir.glob("step_*.stp"):
        try:
            idx = int(p.stem.split("_")[-1])
            max_index = max(max_index, idx)
        except (ValueError, IndexError):
            continue
    return max_index + 1

def find_step_files(root: Path):
    """Recursively find STEP/STP files under the given root."""
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".step",".stp"}:
            files.append(p)
    
    return sorted(files)


def file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file in streaming mode."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_seen_hashes():
    """Load previously processed file hashes from log file."""
    if HASH_LOG.exists():
        return set(HASH_LOG.read_text(encoding="UTF-8").splitlines())
    
    return set()


def save_seen_hashes(hashes):
    """Persist processed hashes after each successful export."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HASH_LOG.write_text("\n".join(sorted(hashes)) + ("\n") if hashes else "", encoding="utf-8")


def export_parts(step_file: Path, out_dir: Path, start_index: int):
    """Export each root shape from one STEP file to individual STP files."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_file))

    if status != IFSelect_RetDone:
        print(f"Okunamadi: {step_file}")
        return 0, start_index
    
    reader.TransferRoots()
    root_count = reader.NbRootsForTransfer()

    if root_count == 0:
        print(f"Bu dosyada parçaya ayrılacak kök yok: {step_file}")
        return 0, start_index
    

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    next_index = start_index
    for i in range(1, root_count + 1):
        reader.TransferRoot(i)
        shape = reader.Shape(i)

        if shape.IsNull():
            continue
        
        # Kalinligi olmayan yüzeyleri filtrele
        if shape.ShapeType() != TopAbs_SOLID:
            print(f"Atlanıyor (solid değil): {step_file} - Root {i}")
            continue

        out_path = out_dir / f"step_{next_index}.stp"
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)

        if writer.Write(str(out_path)) == 1:
            saved += 1
            next_index += 1
            print(f"Kaydedildi: {out_path}")
    return saved, next_index


def main():
    """Split STEP files into per-root STP files with content-based deduplication."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_seen_hashes()
    
    step_files = find_step_files(INPUT_DIR)

    if MAX_FILES is not None:
        step_files = step_files[:MAX_FILES]

    print(f"{len(step_files)} STEP dosyası bulundu.")

    total_saved = 0
    next_step_index = get_next_step_index(OUTPUT_DIR)

    for step_file in step_files:

        hash_value = file_hash(step_file)

        if hash_value in seen_hashes:
            print(f"Atlanıyor (aynı içerik): {step_file}")
            continue

        saved, next_step_index = export_parts(step_file, OUTPUT_DIR, next_step_index)

        if saved > 0:
            seen_hashes.add(hash_value)
            save_seen_hashes(seen_hashes)
            total_saved += saved
    print(f"Toplam kaydedilen parça sayısı: {total_saved}")


if __name__ == "__main__":
    main()