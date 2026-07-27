from pathlib import Path
import requests
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
LIST_URL = "https://deep-geometry.github.io/abc-dataset/data/step_v00.txt"
OUT_DIR = BASE_DIR / "data" / "abc" / "raw" / "step"
MANIFEST_PATH = BASE_DIR / "data" / "abc" / "raw" / "downloaded_chunk.txt"

# Hangi chunk indirilecek:
# 0 -> ilk chunk
CHUNK_INDEX = 0


def download_file(url: str, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024

        with open(outpath, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=outpath.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def main():
    text = requests.get(LIST_URL, timeout=120).text.splitlines()
    lines = [line.strip() for line in text if line.strip()]

    chunk_count = len(lines) // 2
    if CHUNK_INDEX < 0 or CHUNK_INDEX >= chunk_count:
        raise IndexError(f"CHUNK_INDEX aralik disi: 0-{chunk_count - 1}")

    url = lines[CHUNK_INDEX * 2].strip()
    filename = lines[CHUNK_INDEX * 2 + 1].strip()

    if " " in url:
        url = url.split()[0]

    archive_path = OUT_DIR / filename
    download_file(url, archive_path)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write(f"chunk_index={CHUNK_INDEX}\n")
        f.write(f"archive_name={filename}\n")
        f.write(f"archive_url={url}\n")
        f.write(f"archive_path={archive_path}\n")

    print(f"Downloaded: {filename}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()