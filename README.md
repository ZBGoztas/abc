# abc: CAD part similarity search on the ABC dataset (phase 1 baseline)

A baseline for **retrieving similar mechanical parts from STEP files**:
1. Each B-Rep is turned into a fixed-length descriptor.
2. The descriptors are indexed with FAISS.
3. For a query part, the closest parts are returned.

It works on the public [ABC dataset](https://deep-geometry.github.io/abc-dataset/). It is the first phase of the 3DSearch project; later phases replace the hand-made descriptor with a learned B-Rep encoder.

```
STEP ──► orientation + scale normalisation ──► B-Rep features ──► NPZ tensors ──► 256-d descriptor ──► FAISS index
                                                                                                   ▲
query STEP ──► same pipeline ──────────────────────────────────────────────────────────────────────┘ top-k
```

## Pipeline

**1. Get the data** (`download_abc_data/`):

| Script | What it does |
|---|---|
| `abc_install.py` | Downloads one chunk of the STEP archive list |
| `un_abc.py` | Extracts the chunk |
| `ste_file_split.py` | Copies STEP files to `step_N.stp`, skipping duplicates by hash |
| `abc_date_same_data_discard.py` | Moves byte-identical duplicates out of the set |
| `abc_data_split.py` | Small Qt viewer for manually sorting parts into `dataset/` or `rejected/` |

**2. Normalise** (`normalize_pipeline.py`, `src/service/`):
- Align each part to its principal axes of inertia, with a bounding-box asymmetry check to fix the sign of each axis.
- Scale it to a unit bounding box.
- Write the result back to STEP.

**3. Extract features** (`src/service/feature_engine.py`, `tensor_builder.py`), using pythonocc / OpenCASCADE:
- face types (11 classes), face areas and normals sampled on a UV grid;
- edge lengths, face adjacency, bounding box, volume.

These are packed into BRepNet-style NPZ tensors, with 15 features per face and 1 per edge.

**4. Build the descriptor** (`embedding_generator.py`). It concatenates:
- a face-type histogram;
- the mean and standard deviation of the face and edge features;
- the bounding box;
- log volume;
- log face and edge counts.

That gives 49 values, zero-padded to 256 so a learned embedding can replace it later without changing the index format.

**5. Index and search:**
- `run_embedding_pipeline.py` builds `index/faiss.index` (`IndexFlatL2`) and an `id_map.parquet`.
- `find_similar_parts.py <part.step> --top_k 5 --show` normalises the query, embeds it, searches the index and can show the hits in 3D.
- `raw_similarity_pipeline.py` uses each part in `dataset/` as a query against all raw ABC parts, and copies the top-k matches to `raw_similar/`. This is how a small hand-picked set is grown into a larger one.

Metadata (`src/metadata_store.py`) records each part's SHA-256, source path and processing status, so a re-run skips finished parts.

## Limitations

The descriptor is a global statistical summary. It separates parts by size, face mix and complexity, but it ignores how faces are connected, so parts that look alike and are built differently can rank close together. This is the reason for the learned encoder in the later phases.

## Setup

```bash
conda install -c conda-forge pythonocc-core
pip install numpy pandas pyarrow faiss-cpu py7zr requests tqdm PyQt5
python download_abc_data/abc_install.py && python download_abc_data/un_abc.py && python download_abc_data/ste_file_split.py
python normalize_pipeline.py
python run_embedding_pipeline.py
python find_similar_parts.py path/to/query.step --top_k 5 --show
```

All data is written under `data/` and `index/`, which git ignores.
