"""
Embedding Generator (PoC — Basit İstatistiksel Özet Vektör)
-------------------------------------------------------------
config.py'deki embedding_config'e baglidir.
Gorevi:
- tensor_builder.build_tensors() ciktisi olan NPZ tensor sozlugunu alir
- Sabit boyutlu (EMBEDDING_DIM) bir float32 embedding vektoru uretir

ONEMLI: Bu, GERCEK bir ogrenilmis embedding DEGILDIR.
PoC asamasinda EmbeddingStore + FAISS + arama zincirinin
uctan uca calistigini dogrulamak icin elle tasarlanmis
istatistiksel bir ozet vektor uretir.

Faz 2'de bu modul, egitilmis bir Graph Autoencoder'in
encoder cikisini dondurecek sekilde DEGISTIRILECEKTIR.
Bu dosyayi cagiran kodun (orn. pipeline orchestrator)
sadece bu fonksiyonu import ettigi surece, ust katmanlarda
degisiklik gerekmeyecektir.
"""

from typing import Dict
import numpy as np

from src.config import embedding_config, npz_schema_config
from src.braketLog.logger import get_logger

logger = get_logger(__name__)


# tensor_builder ile birebir uyumlu olmali (face_features one-hot boyutu)
NUM_FACE_TYPES = 11


def generate_embedding(
    tensors: Dict[str, np.ndarray],
    part_id: str = "unknown",
    embedding_dim: int = None,
) -> np.ndarray:
    """
    NPZ tensor sozlugunden sabit boyutlu bir embedding vektoru uretir.

    Parametreler
    ------------
    tensors : Dict[str, np.ndarray]
        tensor_builder.build_tensors() ciktisi (veya NpzStore'dan
        geri okunmus ayni sekilli sozluk).
    part_id : str
        Sadece loglama amacli.
    embedding_dim : int, optional
        Verilmezse embedding_config.EMBEDDING_DIM kullanilir.

    Donus
    -----
    np.ndarray, shape [embedding_dim], dtype float32
    """
    embedding_dim = embedding_dim or embedding_config.EMBEDDING_DIM

    try:
        face_features = np.asarray(tensors["face_features"], dtype=np.float32)   # [F, 15]
        edge_features = np.asarray(tensors["edge_features"], dtype=np.float32)   # [E, 1]
        face_types = np.asarray(tensors["face_types"], dtype=np.int64)           # [F]
        bbox = np.asarray(tensors["bbox"], dtype=np.float32).reshape(-1)         # [3]
        volume = np.asarray(tensors["volume"], dtype=np.float64).reshape(-1)     # [1]
    except KeyError as e:
        logger.error(f"Embedding uretimi icin zorunlu tensor eksik: {e} | part_id={part_id}")
        raise

    F = face_features.shape[0]
    E = edge_features.shape[0]

    if F == 0:
        logger.error(f"face_features bos, embedding uretilemiyor | part_id={part_id}")
        raise ValueError(f"face_features bos: part_id={part_id}")

    # 1) face_types histogrami (normalize) -> [NUM_FACE_TYPES]
    face_type_hist = np.zeros(NUM_FACE_TYPES, dtype=np.float32)
    counts = np.bincount(face_types, minlength=NUM_FACE_TYPES)[:NUM_FACE_TYPES]
    face_type_hist = counts.astype(np.float32) / max(F, 1)

    # 2) face_features ortalama ve std -> [15] + [15]
    face_feat_mean = face_features.mean(axis=0)
    face_feat_std = face_features.std(axis=0)

    # 3) edge_features ortalama ve std -> [1] + [1]
    if E > 0:
        edge_feat_mean = edge_features.mean(axis=0)
        edge_feat_std = edge_features.std(axis=0)
    else:
        logger.warning(f"edge_features bos, sifir ile dolduruluyor | part_id={part_id}")
        edge_feat_mean = np.zeros(edge_features.shape[1] if edge_features.ndim > 1 else 1, dtype=np.float32)
        edge_feat_std = np.zeros_like(edge_feat_mean)

    # 4) bbox -> [3] (zaten normalize edilmis halde geliyor)
    bbox_vec = bbox

    # 5) volume -> log-scale [1] (buyuk hacim farklarini yumusatmak icin)
    volume_log = np.log1p(np.abs(volume)).astype(np.float32)

    # 6) topolojik karmasiklik gostergesi -> [2] (F, E; log-scale, sinirsiz buyumeye karsi)
    topo_complexity = np.array(
        [np.log1p(F), np.log1p(E)],
        dtype=np.float32,
    )

    # ---- Tum parcalari birlestir ----
    summary_vector = np.concatenate([
        face_type_hist,     # 11
        face_feat_mean,     # 15
        face_feat_std,      # 15
        edge_feat_mean,     # 1
        edge_feat_std,      # 1
        bbox_vec,           # 3
        volume_log,         # 1
        topo_complexity,    # 2
    ]).astype(np.float32)

    summary_dim = summary_vector.shape[0]

    if summary_dim > embedding_dim:
        logger.error(
            f"Ozet vektor boyutu ({summary_dim}) EMBEDDING_DIM'den ({embedding_dim}) "
            f"buyuk, kirpma yapilamaz | part_id={part_id}"
        )
        raise ValueError(
            f"Ozet vektor boyutu ({summary_dim}) EMBEDDING_DIM ({embedding_dim}) icine sigmiyor."
        )

    # Geri kalan kismi sifir ile doldur (Faz 2'de ogrenilmis embedding ile degisecek)
    embedding = np.zeros(embedding_dim, dtype=np.float32)
    embedding[:summary_dim] = summary_vector

    logger.debug(
        f"Embedding uretildi (PoC-ozet): part_id={part_id}, "
        f"ozet_boyut={summary_dim}, toplam_boyut={embedding_dim}, F={F}, E={E}"
    )

    return embedding
