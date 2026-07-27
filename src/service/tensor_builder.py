"""
Tensor Builder (FeatureEngine -> BRepNet NPZ tensor formati)
---------------------------------------------------------------
Gorevi: FeatureEngine'in ham ciktisini (topoloji, yuzey tipleri,
normaller, fiziksel ozellikler) alip, config.py'deki
NpzSchemaConfig.KEYS ile birebir uyumlu, sabit sekilli
(fixed-shape) numpy tensorlerine donusturur.

Tasarim kararlari (kullanici ile netlestirildi):
- G (normal grid boyutu) = 25 sabit (curved yuzeylerde zaten
  grid_size=5x5=25 uretiliyor; planar yuzeylerde tek normal
  25 kez tekrarlanarak G=25'e tamamlanir).
- face_features = [tip one-hot (11)] + [ortalama normal (3)] +
  [normalize yuzey alani (1)] = 15 boyut.
- edge_features = [normalize kenar uzunlugu (1)] = 1 boyut
  (NPZ schema'nin REQUIRED_KEYS'i icin minimum gecerli tensor).
"""

from typing import Dict
import numpy as np

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from src.braketLog.logger import get_logger

logger = get_logger(__name__)

# Yuzey tipi kategori sirasi -- yuzey_tipleri_tespit_et() sozlugunun
# anahtar sirasiyla BIREBIR ayni olmali (one-hot index tutarliligi icin).
FACE_TYPE_CATEGORIES = [
    "duzYuzeyler",
    "silindirikYuzeyler",
    "koniYuzeyler",
    "kureYuzeyler",
    "torusYuzeyler",
    "BSplineYuzeyler",
    "BezierYuzeyler",
    "devirYuzeyler",
    "SurfaceOfExtrusion",
    "offsetYuzeyler",
    "digerYuzeyler",
]
NUM_FACE_TYPES = len(FACE_TYPE_CATEGORIES)  # 11
NORMAL_GRID_SIZE = 25  # G sabiti (kullanici karari)


def _yuzey_alani_hesapla(face) -> float:
    """Tek bir yuzeyin alanini GProp_GProps ile hesaplar."""
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def _kenar_uzunlugu_hesapla(edge) -> float:
    """Tek bir kenarin uzunlugunu GProp_GProps ile hesaplar."""
    props = GProp_GProps()
    brepgprop.LinearProperties(edge, props)
    return props.Mass()


def _face_tip_index_haritasi(yuzey_tipleri_dict: dict, base_name: str) -> Dict[int, int]:
    """
    Her yuzey nesnesinin (id() bazli) hangi tip kategori index'ine
    ait oldugunu tutan bir harita olusturur.

    NOT: id() kullanimi, topoloji_gez() cache'inden gelen ayni face
    referanslari kullanildigi surece guvenlidir (FeatureEngine'deki
    yuzey_normalleri() ile ayni varsayim).
    """
    harita = {}
    for kategori_idx, kategori_adi in enumerate(FACE_TYPE_CATEGORIES):
        adet = len(yuzey_tipleri_dict[kategori_adi])
        if adet > 0:
            logger.debug(f"{base_name}: '{kategori_adi}' -> {adet} yuzey (tip_idx={kategori_idx})")
        for face in yuzey_tipleri_dict[kategori_adi]:
            harita[id(face)] = kategori_idx
    logger.debug(f"{base_name}: Face tip index haritasi olusturuldu. Toplam eslesme={len(harita)}")
    return harita


def _normalleri_sabit_grid_boyutuna_getir(normal_kaydi: dict, base_name: str, face_idx: int) -> tuple:
    """
    Bir yuzeyin (planar veya curved) normal listesini, sabit
    NORMAL_GRID_SIZE (=25) uzunluguna getirir.

    - Curved yuzeyler zaten grid_size=5x5=25 uretiyor -> oldugu gibi kullanilir.
    - Planar yuzeyler tek normal uretiyor -> o normal 25 kez tekrarlanir.
    - normal_kaydi None ise (hesaplanamayan yuzey) sifir vektorlerle doldurulur
      ve mask tamamen False olur.

    :return: (normals_array [G,3], mask_array [G])
    """
    if normal_kaydi is None:
        logger.warning(
            f"{base_name}: face_idx={face_idx} icin normal kaydi None, "
            f"sifir vektor + bos mask kullanilacak"
        )
        normals = np.zeros((NORMAL_GRID_SIZE, 3), dtype="float32")
        mask = np.zeros((NORMAL_GRID_SIZE, ), dtype="bool")
        return normals, mask
    
    normaller = normal_kaydi["normals"]
    mevcut_sayi = len(normaller)

    if mevcut_sayi == NORMAL_GRID_SIZE:
        normals = np.asarray(normaller, dtype="float32")
        mask = np.ones((NORMAL_GRID_SIZE, ), dtype="bool")
        return normals, mask
    
    if mevcut_sayi == 1:
        # Planar yuzey: tek normal 25 kez tekrarlanir (gecerli, sabit deger).
        tekrarlanan = np.tile(np.asarray(normaller[0], dtype="float32"), (NORMAL_GRID_SIZE, 1))
        mask = np.ones((NORMAL_GRID_SIZE, ), dtype="bool")
        return tekrarlanan, mask
    
    # Beklenmeyen bir durum (grid_size farkli calistirilmis olabilir):
    # mevcut kadarini yaz, kalanini sifirla ve mask ile isaretle.
    logger.warning(
        f"{base_name}: face_idx={face_idx} icin beklenmeyen normal sayisi={mevcut_sayi} "
        f"(beklenen 1 veya {NORMAL_GRID_SIZE}), kismi doldurma yapilacak"
    )

    normals = np.zeros((NORMAL_GRID_SIZE, 3), dtype="float32")
    mask = np.zeros((NORMAL_GRID_SIZE,), dtype="bool")
    sinir = min(mevcut_sayi, NORMAL_GRID_SIZE)
    normals[:sinir] = np.asarray(normaller[:sinir], dtype="float32")
    mask[:sinir] = True
    return normals, mask

def build_tensors(feature_engine, grid_size: int = 5) -> Dict[str, np.ndarray]:
    """
    FeatureEngine'in ham ciktisini, NpzSchemaConfig.KEYS ile uyumlu
    sabit sekilli tensorlere donusturur.

    :param feature_engine: Ilgili parca icin olusturulmus FeatureEngine ornegi
    :param grid_size: yuzey_normalleri() icin kullanilan grid boyutu (varsayilan 5 -> 5x5=25)
    :return: NpzStore.save() icin hazir tensors sozlugu
    """

    base_name = feature_engine.baseName
    logger.debug(f"{base_name}: Tensor olusturma islemi basladi (grid_size={grid_size})")

    try:
        topoloji = feature_engine.topoloji_gez()
        yuzey_tipleri = feature_engine.yuzey_tipleri_tespit_et()
        yuzey_normalleri = feature_engine.yuzey_normalleri(grid_size = grid_size)
        bbox = feature_engine.bounding_box()
        fiziksel = feature_engine.physical_features()

        if None in (topoloji, yuzey_tipleri, yuzey_normalleri, bbox, fiziksel):
            logger.error(
                f"{base_name}: FeatureEngine'den gelen kaynak verilerden biri None, "
                f"tensor olusturulamiyor (topoloji={topoloji is not None}, "
                f"yuzeyTipleri={yuzey_tipleri is not None}, "
                f"yuzeyNormalleri={yuzey_normalleri is not None}, "
                f"bbox={bbox is not None}, fiziksel={fiziksel is not None})"
            )
            raise ValueError(f"{base_name}: Tensor icin gerekli FeatureEngine ciktilari eksik")
        

        yuzeyler = topoloji["faces"]
        kenarlar = topoloji["edges"]
        F = len(yuzeyler)
        E = len(kenarlar)

        logger.debug(f"{base_name}: Topoloji okundu. F={F}, E={E}")
        tip_haritasi = _face_tip_index_haritasi(yuzey_tipleri, base_name)

        # --- BBOX DIAGONAL (kenar uzunluklarini normalize etmek icin referans) ---
        bbox_diagonal = float(np.sqrt(bbox["x"] ** 2 + bbox["y"] ** 2 + bbox["z"] ** 2))
        if bbox_diagonal <=1e-9:
            logger.warning(f"{base_name}: bbox diagonal ~0, normalize icin 1.0 kullanilacak")
            bbox_diagonal = 1.0
        else:
            logger.debug(f"{base_name}: bbox diagonal={bbox_diagonal:.6f}")

        # --- YUZEY ALANLARI (normalize icin toplam alan referansi) ---
        yuzey_alanlari = [_yuzey_alani_hesapla(face) for face in yuzeyler]
        toplam_alan = sum(yuzey_alanlari)

        if toplam_alan <= 1e-9:
            logger.warning(f"{base_name}: toplam yuzey alani ~0, normalize icin 1.0 kullanilacak")
            toplam_alan = 1.0
        else:
            logger.debug(f"{base_name}: toplam yuzey alani={toplam_alan:.6f}")

        # --- FACE_FEATURES [F, 15] VE FACE_TYPES [F] VE NORMALS [F,25,3] / MASK [F,25] ---
        face_feature_dim = NUM_FACE_TYPES + 3 + 1
        face_features = np.zeros((F, face_feature_dim), dtype="float32")
        face_types = np.zeros((F, ), dtype="int16")
        normals_arr = np.zeros((F, NORMAL_GRID_SIZE, 3), dtype="float32")
        normals_mask = np.zeros((F, NORMAL_GRID_SIZE), dtype="bool")

        for i, face in enumerate(yuzeyler):
            tip_idx = tip_haritasi.get(id(face), NUM_FACE_TYPES - 1)
            if id(face) not in tip_haritasi:
                logger.warning(f"{base_name}: face_idx={i} tip haritasinda bulunamadi, 'diger' olarak isaretlendi")
            face_types[i] = tip_idx

            one_hot = np.zeros((NUM_FACE_TYPES,), dtype="float32")
            one_hot[tip_idx] = 1.0

            normal_kaydi = yuzey_normalleri[i]
            n_grid, mask = _normalleri_sabit_grid_boyutuna_getir(normal_kaydi, base_name, i)
            normals_arr[i] = n_grid
            normals_mask[i] = mask

            if mask.any():
                ortalama_normal = n_grid[mask].mean(axis=0)
            else:
                ortalama_normal = np.zeros((3,), dtype="float32")

            normalize_alan = float(yuzey_alanlari[i] / toplam_alan)
            face_features[i] = np.concatenate([one_hot, ortalama_normal, [normalize_alan]])
        logger.info(f"{base_name}: face_features, face_types, normals, normals_mask olusturuldu. shape(face_features)={face_features.shape}")

        # --- EDGE_FEATURES [E, 1] (normalize kenar uzunlugu) ---
        edge_features = np.zeros((E, 1), dtype="float32")
        for i, edge in enumerate(kenarlar):
            uzunluk = _kenar_uzunlugu_hesapla(edge)
            edge_features[i, 0] = uzunluk / bbox_diagonal


        logger.info(f"{base_name}: edge_features olusturuldu. shape={edge_features.shape}")

        # --- EDGE_TO_FACE [E, 2] (eksik olan -1 ile doldurulur) ---
        edge_to_face = np.full((E, 2), fill_value=-1, dtype="int32")
        eksik_komsuluk_sayisi = 0

        for edge_idx, komsu_yuzeyler in topoloji["adjacency_edge_to_face"].items():
            if len(komsu_yuzeyler) < 2:
                eksik_komsuluk_sayisi += 1
            for slot, face_idx in enumerate(komsu_yuzeyler[:2]):
                edge_to_face[edge_idx, slot] = face_idx

        if eksik_komsuluk_sayisi > 0:
            logger.debug(
                f"{base_name}: {eksik_komsuluk_sayisi} kenar tek yuzeye bagli "
                f"(sinir/acik kenar), -1 ile dolduruldu"
            )
        logger.info(f"{base_name}: edge_to_face olusturuldu. shape={edge_to_face.shape}")

        # --- FACE_ADJACENCY_INDEX [2, K] (yonlu kenar listesi formatinda) ---
        kaynak_liste = []
        hedef_liste = []
        for face_idx, komsu_yuzeyler in topoloji["adjacency_face_to_face"].items():
            for komsu_idx in komsu_yuzeyler:
                kaynak_liste.append(face_idx)
                hedef_liste.append(komsu_idx)

        if kaynak_liste:
            face_adjacency_index = np.array([kaynak_liste, hedef_liste], dtype="int32")
        else:
            logger.warning(f"{base_name}: face_adjacency_index bos (K=0), izole yuzeyler olabilir")
            face_adjacency_index = np.zeros((2, 0), dtype="int32")

        
        logger.info(f"{base_name}: face_adjacency_index olusturuldu. K={face_adjacency_index.shape[1]}")

        # --- BBOX / CENTER / VOLUME ---
        bbox_array = np.array([bbox["x"], bbox["y"], bbox["z"]], dtype="float32")
        center_array = np.array(bbox["merkez"], dtype="float32")
        volume_array = np.array([fiziksel["hacim"]], dtype="float64")

        tensors = {
            "face_features": face_features,
            "edge_features": edge_features,
            "edge_to_face": edge_to_face,
            "face_adjacency_index": face_adjacency_index,
            "face_types": face_types,
            "normals": normals_arr,
            "normals_mask": normals_mask,
            "bbox": bbox_array,
            "center": center_array,
            "volume": volume_array,
        }

        logger.info(
            f"{base_name}: Tum tensorler basari ile olusturuldu. "
            f"F={F}, E={E}, K={face_adjacency_index.shape[1]}"
        )
        return tensors
    except Exception as e:
        logger.exception(f"{base_name}: Tensor olusturulurken hata olustu")
        raise