"""
main.py
--------
PoC Pipeline - Adim 1 + Adim 2:
    STEP oku -> Orientation Normalize et -> kaydet
             -> Scale Normalize et       -> kaydet

Once TEK bir dosya uzerinde (break ile) test edilir, sorun yoksa
break kaldirilarak tum klasore uygulanir.
"""

from pathlib import Path
from src.config import Step_Data_Path, BASE_DIR
from src.service.step_file_processes import stepReader, shape_to_step
from src.service.orientation_normalizasyon import atalet_eksenlerine_gore_normalizasyon
from src.service.scale_normalize import olcek_normalizasyonu
from src.braketLog.logger import get_logger

logger = get_logger(__name__)

# Normalize edilmis STEP dosyalarinin kaydedilecegi klasorler
ORYANTASYON_NORMALIZED_STEP_DIR = BASE_DIR / "data" / "oryantasyon_normalized_step"
ORYANTASYON_NORMALIZED_STEP_DIR.mkdir(parents=True, exist_ok=True)

ORYANTASYON_AND_SCALE_NORMALIZED_STEP_DIR = BASE_DIR / "data" / "oryantasyon_and_scale_normalized_step"
ORYANTASYON_AND_SCALE_NORMALIZED_STEP_DIR.mkdir(parents=True, exist_ok=True)


def process_single_file(step_path: Path) -> bool:
    """
    Tek bir STEP dosyasi icin:
        oku -> orientation normalize et -> kaydet
            -> scale normalize et       -> kaydet
    Basarili olursa True, herhangi bir adimda basarisiz olursa False doner.
    """

    file_name = step_path.name

    # --- 1) STEP OKU ---
    shape = stepReader(str(step_path))
    if shape is None:
        logger.error(f"{file_name}: Shape okunamadi, pipeline durduruldu (bu dosya icin)")
        return False

    # --- 2) ORIENTATION NORMALIZE ET ---
    oryantasyon_shape = atalet_eksenlerine_gore_normalizasyon(file_name, shape)
    if oryantasyon_shape is None or oryantasyon_shape.IsNull():
        logger.error(f"{file_name}: Orientation normalize edilmis shape gecersiz, kayit atlandi")
        return False

    # --- 3) ORIENTATION NORMALIZE EDILMIS SHAPE'I KAYDET ---
    oryantasyon_out_path = ORYANTASYON_NORMALIZED_STEP_DIR / file_name
    success = shape_to_step(file_name, oryantasyon_shape, str(oryantasyon_out_path))
    if not success:
        logger.error(f"{file_name}: Orientation normalize edilmis STEP kaydedilemedi")
        return False
    logger.info(f"{file_name}: Adim 1 (orientation normalize) basari ile tamamlandi -> {oryantasyon_out_path}")

    # --- 4) SCALE NORMALIZE ET ---
    scale_shape = olcek_normalizasyonu(file_name, oryantasyon_shape)
    if scale_shape is None or scale_shape.IsNull():
        logger.error(f"{file_name}: Scale normalize edilmis shape gecersiz, kayit atlandi")
        return False

    # --- 5) SCALE NORMALIZE EDILMIS SHAPE'I KAYDET ---
    scale_out_path = ORYANTASYON_AND_SCALE_NORMALIZED_STEP_DIR / file_name
    success = shape_to_step(file_name, scale_shape, str(scale_out_path))
    if not success:
        logger.error(f"{file_name}: Scale normalize edilmis STEP kaydedilemedi")
        return False

    logger.info(f"{file_name}: Adim 2 (scale normalize) basari ile tamamlandi -> {scale_out_path}")
    return True


def main():
    step_files = sorted(
        list(Step_Data_Path.glob("*.step")) + list(Step_Data_Path.glob("*.stp"))
    )

    if not step_files:
        logger.error(f"STEP dosyasi bulunamadi: {Step_Data_Path}")
        return

    logger.info(f"Toplam {len(step_files)} STEP dosyasi bulundu, isleme baslaniyor")

    basarili_sayisi = 0
    basarisiz_sayisi = 0

    for step_path in step_files:
        logger.info(f"Isleniyor: {step_path.name}")

        sonuc = process_single_file(step_path)
        if sonuc:
            basarili_sayisi += 1
        else:
            basarisiz_sayisi += 1

        # --- TEK DOSYA TEST MODU ---
        # Ilk calistirmada sadece 1 dosya islensin diye break kullaniyoruz.
        # Sonuclar dogrulandiktan sonra bu satiri kaldirip tum klasoru islet.

    logger.info(f"Islem tamamlandi | Basarili: {basarili_sayisi} | Basarisiz: {basarisiz_sayisi}")


if __name__ == "__main__":
    main()
