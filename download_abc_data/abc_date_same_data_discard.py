from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STEP_DIR = BASE_DIR / "data" / "abc" / "raw" / "step_data"
DEFAULT_DUPLICATES_DIR = BASE_DIR / "data" / "abc" / "rejected" / "duplicates_step_data"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="step_data icindeki tekrar STEP dosyalarini tespit eder ve benzersizleri birakir."
	)
	parser.add_argument(
		"--step-dir",
		type=Path,
		default=DEFAULT_STEP_DIR,
		help=f"Taranacak klasor (varsayilan: {DEFAULT_STEP_DIR})",
	)
	parser.add_argument(
		"--duplicates-dir",
		type=Path,
		default=DEFAULT_DUPLICATES_DIR,
		help=(
			"Tekrar dosyalarin tasinacagi klasor "
			f"(varsayilan: {DEFAULT_DUPLICATES_DIR})"
		),
	)
	parser.add_argument(
		"--delete",
		action="store_true",
		help="Tekrar dosyalari tasimak yerine kalici olarak sil.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Islem yapmadan sadece hangi dosyalarin tekrar oldugunu yazdir.",
	)
	return parser.parse_args()


def iter_step_files(root: Path) -> list[Path]:
	return sorted(
		p
		for p in root.rglob("*")
		if p.is_file() and p.suffix.lower() in {".stp", ".step"}
	)


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(chunk_size), b""):
			h.update(chunk)
	return h.hexdigest()


def load_step_vertices(path: Path) -> np.ndarray | None:
	reader = STEPControl_Reader()
	status = reader.ReadFile(str(path))
	if status != IFSelect_RetDone:
		print(f"Uyari: STEP okunamadi, geometrik karsilastirma atlandi -> {path}")
		return None

	reader.TransferRoots()
	shape = reader.OneShape()
	if shape.IsNull():
		print(f"Uyari: STEP shape gecersiz, geometrik karsilastirma atlandi -> {path}")
		return None

	mesh = BRepMesh_IncrementalMesh(shape, 0.5, False, 0.5, True)
	mesh.Perform()
	if not mesh.IsDone():
		print(f"Uyari: Mesh olusturulamadi, geometrik karsilastirma atlandi -> {path}")
		return None

	vertices: list[list[float]] = []
	explorer = TopExp_Explorer(shape, TopAbs_FACE)
	while explorer.More():
		face = explorer.Current()
		location = TopLoc_Location()
		triangulation = BRep_Tool.Triangulation(face, location)

		if triangulation is not None:
			for i in range(1, triangulation.NbNodes() + 1):
				node = triangulation.Node(i)
				if not location.IsIdentity():
					node = node.Transformed(location.Transformation())
				vertices.append([node.X(), node.Y(), node.Z()])

		explorer.Next()

	if not vertices:
		print(f"Uyari: Nokta cikarilamadi, geometrik karsilastirma atlandi -> {path}")
		return None

	return np.asarray(vertices, dtype=np.float64)


def normalize_orientation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	centered = points - points.mean(axis=0)

	if len(centered) < 3:
		return centered, np.zeros(3, dtype=np.float64)

	cov = np.cov(centered, rowvar=False)
	eigvals, eigvecs = np.linalg.eigh(cov)
	order = np.argsort(eigvals)[::-1]
	eigvals = eigvals[order]
	eigvecs = eigvecs[:, order]

	if np.linalg.det(eigvecs) < 0:
		eigvecs[:, 2] *= -1

	aligned = centered @ eigvecs

	# Eksen sign belirsizligini stabil hale getir.
	for axis in range(3):
		moment3 = float(np.sum(aligned[:, axis] ** 3))
		if abs(moment3) < 1e-12:
			pivot_idx = int(np.argmax(np.abs(aligned[:, axis])))
			flip = aligned[pivot_idx, axis] < 0
		else:
			flip = moment3 < 0

		if flip:
			aligned[:, axis] *= -1

	return aligned, eigvals


def geometric_signature(path: Path) -> tuple[str, tuple[float, float, float], tuple[float, float, float]] | None:
	points = load_step_vertices(path)
	if points is None:
		return None

	points = points[np.isfinite(points).all(axis=1)]
	if len(points) == 0:
		return None

	aligned, eigvals = normalize_orientation(points)
	extents = np.sort(aligned.max(axis=0) - aligned.min(axis=0))
	radii = np.sort(np.linalg.norm(aligned, axis=1))

	if len(radii) > 6000:
		indices = np.linspace(0, len(radii) - 1, 6000, dtype=np.int64)
		radii = radii[indices]

	radii_quantized = np.round(radii, 5).astype(np.float32)
	radial_hash = hashlib.sha256(radii_quantized.tobytes()).hexdigest()

	eigvals_rounded = tuple(np.round(eigvals, 8).tolist())
	extents_rounded = tuple(np.round(extents, 6).tolist())

	return radial_hash, eigvals_rounded, extents_rounded


def geometry_close(
	a_eigvals: tuple[float, float, float],
	b_eigvals: tuple[float, float, float],
	a_extents: tuple[float, float, float],
	b_extents: tuple[float, float, float],
) -> bool:
	return np.allclose(a_eigvals, b_eigvals, rtol=1e-4, atol=1e-7) and np.allclose(
		a_extents, b_extents, rtol=1e-4, atol=1e-5
	)


def unique_destination_path(target_dir: Path, source_name: str) -> Path:
	target = target_dir / source_name
	if not target.exists():
		return target

	stem = Path(source_name).stem
	suffix = Path(source_name).suffix
	i = 1
	while True:
		candidate = target_dir / f"{stem}__dup{i}{suffix}"
		if not candidate.exists():
			return candidate
		i += 1


def deduplicate_step_files(
	step_dir: Path,
	duplicates_dir: Path,
	delete_duplicates: bool,
	dry_run: bool,
) -> int:
	if not step_dir.exists():
		print(f"Hata: Klasor bulunamadi -> {step_dir}")
		return 1

	files = iter_step_files(step_dir)
	if not files:
		print(f"STEP dosyasi bulunamadi: {step_dir}")
		return 0

	print(f"Toplam {len(files)} STEP dosyasi bulundu.")

	seen_hashes: dict[str, Path] = {}
	seen_geometric: dict[str, tuple[Path, tuple[float, float, float], tuple[float, float, float]]] = {}
	duplicates: list[tuple[Path, Path]] = []

	for path in files:
		digest = file_hash(path)
		if digest in seen_hashes:
			duplicates.append((path, seen_hashes[digest]))
			continue

		seen_hashes[digest] = path

		geo_sig = geometric_signature(path)
		if geo_sig is None:
			continue

		radial_hash, eigvals, extents = geo_sig
		if radial_hash in seen_geometric:
			original, original_eigvals, original_extents = seen_geometric[radial_hash]
			if geometry_close(eigvals, original_eigvals, extents, original_extents):
				duplicates.append((path, original))
				continue

		seen_geometric[radial_hash] = (path, eigvals, extents)

	if not duplicates:
		print("Tekrar dosya bulunamadi. Tum dosyalar benzersiz.")
		return 0

	print(f"{len(duplicates)} adet tekrar dosya bulundu.")
	for dup, original in duplicates:
		print(f"TEKRAR: {dup} | ORJINAL: {original}")

	if dry_run:
		print("Dry-run aktif: hicbir dosyada degisiklik yapilmadi.")
		return 0

	if delete_duplicates:
		for dup, _original in duplicates:
			dup.unlink()
		print(f"{len(duplicates)} tekrar dosya kalici olarak silindi.")
		return 0

	duplicates_dir.mkdir(parents=True, exist_ok=True)
	for dup, _original in duplicates:
		target = unique_destination_path(duplicates_dir, dup.name)
		shutil.move(str(dup), str(target))

	print(
		f"{len(duplicates)} tekrar dosya {duplicates_dir} klasorune tasindi. "
		"Kaynak klasorde yalnizca benzersiz dosyalar kaldi."
	)
	return 0


def main() -> int:
	args = parse_args()
	return deduplicate_step_files(
		step_dir=args.step_dir,
		duplicates_dir=args.duplicates_dir,
		delete_duplicates=args.delete,
		dry_run=args.dry_run,
	)


if __name__ == "__main__":
	raise SystemExit(main())
