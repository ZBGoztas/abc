'''
NORMALIZE EDILMIS (ATALET EKSENLERINE HIZALANMIS VE OLCEKLENMIS) SHAPE'I
STEP FORMATINDA DISK'E KAYDEDER
'''
import os

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.TopoDS import TopoDS_Shape
import os
from src.braketLog.logger import get_logger
logger = get_logger(__name__)


def shape_to_step(
        fileName: str,
        shape: TopoDS_Shape,
        outpath: str,
        step_schema: str = "AP214"
) -> bool:
    """
    Verilen TopoDS_Shape'i STEP formatinda diske kaydeder.

    :param fileName: Loglama icin dosya/parca adi (orijinal kaynak dosya adi)
    :param shape: Kaydedilecek TopoDS_Shape (orn. normalize edilmis shape)
    :param outpath: STEP dosyasinin tam yolu (orn. "/output/parca.step")
    :param step_schema: STEP semasi ("AP203" veya "AP214"). Varsayilan "AP214"
                         (renk, katman gibi ek bilgileri destekler).
    :return: Kayit basariliysa True, basarisizsa False.
    """
    try:
        if shape is None or shape.IsNull():
            logger.error(f"{fileName}: Kaydedilecek shape gecersiz (null), STEP yazilamiyor")
            return False

        # NOT: outpath DEGISKENINI EZMIYORUZ, KLASOR YOLUNU AYRI DEGISKENDE TUTUYORUZ
        cikis_klasoru = os.path.dirname(outpath)
        if cikis_klasoru and not os.path.exists(cikis_klasoru):
            os.makedirs(cikis_klasoru, exist_ok=True)
            logger.info(f"{fileName}: Cikis klasoru olusturuldu: {cikis_klasoru}")

        # STEP SEMASINI AYARLA (AP203 / AP214)
        Interface_Static.SetCVal("write.step.schema", step_schema)

        # STEP WRITER OLUSTUR VE SHAPE'I EKLE
        writer = STEPControl_Writer()
        durum_transfer = writer.Transfer(shape, STEPControl_AsIs)

        if durum_transfer != IFSelect_RetDone:
            logger.error(f"{fileName}: Shape STEP writer'a transfer edilemedi "
                         f"(durum kodu: {durum_transfer})")
            return False

        # DOSYAYA YAZ (TAM DOSYA YOLU: outpath)
        durum_yazma = writer.Write(outpath)

        if durum_yazma != IFSelect_RetDone:
            logger.error(f"{fileName}: STEP dosyasi diske yazilamadi "
                         f"(durum kodu: {durum_yazma}) -> {outpath}")
            return False

        logger.info(f"{fileName}: STEP dosyasi basariyla kaydedildi -> {outpath}")
        return True

    except Exception:
        logger.exception(f"{fileName}: STEP dosyasi kaydedilirken hata olustu -> {outpath}")
        return False
    
def stepReader(filePath):
    try:
        baseName = os.path.basename(filePath)
        reader = STEPControl_Reader()
        status = reader.ReadFile(filePath)

        if status != IFSelect_RetDone:
            raise Exception(f"Step dosyasi okunamadi: {status}")
        reader.TransferRoots()
        shape = reader.OneShape()

        logger.info(f"Step dosyasi basari ile yuklendi: {baseName}")
        return shape

    except Exception:
        logger.exception(f"Step dosyasi okunurken hata olustu: {baseName}")
        return None
