'''
MODELIN BOUNDING BOX'INI, ASPECT RATIO'YU BOZMADAN (UNIFORM SCALE),
-0.95 ile 0.95 ARALIGINA SIGACAK SEKILDE OLCEKLENDIRIR (BIRIM KUP NORMALIZASYONU)
'''

from OCC.Core.gp import gp_Trsf, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import TopoDS_Shape

from src.braketLog.logger import get_logger
logger = get_logger(__name__)

def _bounding_box_al(shape: TopoDS_Shape) -> tuple:
    """Shape'in bounding box koordinatlarini (xmin,xmax,ymin,ymax,zmin,zmax) dondurur."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, xmax, ymin, ymax, zmin, zmax

def olcek_normalizasyonu(
        fileName: str,
        shape: TopoDS_Shape,
        hedef_sinir: float = 0.95,
    ) -> TopoDS_Shape:
    """
    Modelin bounding box'ini, aspect ratio'yu (en/boy/yukseklik oranini)
    BOZMADAN, tek bir uniform scale factor ile [-hedef_sinir, +hedef_sinir]
    araligina sigacak sekilde olcekler.

    Neden uniform (tek eksende degil, XYZ ayni oranda) scale yapiyoruz:
    - Farkli eksenlerde farkli scale uygulamak (non-uniform scale) modelin
      gercek sekil/oranlarini bozar (kup dikdortgene, kure elipsoide donusur).
    - Bu yuzden en 'uzun' eksene (en buyuk mutlak koordinat degerine) bakip
      TUM eksenleri o degere gore aynı faktorle olcekliyoruz.

    hedef_sinir=0.95 secilmesinin nedeni:
    - Tam [-1, 1] araligina sigdirmak, kayan nokta (floating point) hatalari,
      mesh/tesselasyon sapmalari veya sonraki islemlerdeki (rotasyon,
      export, tolerans) kucuk tasmalar yuzunden modelin sinirdan disari
      tasmasina (>1.0) sebep olabilir. 0.95 sinirinin biraktigi %5'lik
      bosluk, bu tasmalara karsi guvenlik payi (tolerans) saglar.

    NOT: Bu fonksiyon modelin (0,0,0) merkezinde/civarinda oldugunu
    varsayar (orn. atalet eksenlerine gore normalizasyon sonrasi). Model
    merkezde degilse, olcekleme sonrasi bounding box hedef araliga tam
    simetrik oturmayabilir; gerekirse once merkezleme (centering) yapin.

    :param fileName: Loglama icin dosya adi
    :param shape: Olceklendirilecek TopoDS_Shape
    :param hedef_sinir: Bounding box'in sigmasi istenen mutlak sinir (varsayilan 0.95)
    :return: Olceklendirilmis TopoDS_Shape. Hata durumunda orijinal shape dondurulur.
    """

    try:
        xmin, xmax, ymin, ymax, zmin, zmax = _bounding_box_al(shape)

        # Her eksendeki mutlak en buyuk deger (merkez orijinde degilse de
        # en kotu durumu (worst-case) kapsar)
        maks_mutlak_deger = max(
            abs(xmin), abs(xmax),
            abs(ymin), abs(ymax),
            abs(zmin), abs(zmax),
        )

        if maks_mutlak_deger <= 1e-9:
            logger.warning(f"{fileName}: Bounding box boyutu ~0, olcekleme atlaniyor")
            return shape
    
        # Uniform scale factor: modelin en uzak noktasi, hedef_sinir
        # degerine tam denk gelecek sekilde hesaplanir.
        # Ornek: en_uzak_nokta=100, hedef_sinir=0.95 -> factor = 0.95/100 = 0.0095

        scale_factor = hedef_sinir / maks_mutlak_deger

        logger.info(
            f"{fileName}: Bounding box [x:{xmin:.3f},{xmax:.3f}] "
            f"[y:{ymin:.3f},{ymax:.3f}] [z:{zmin:.3f},{zmax:.3f}] | "
            f"maks_mutlak_deger={maks_mutlak_deger:.3f} | scale_factor={scale_factor:.6f}"
        )

        # Olcekleme merkezi olarak orijin (0,0,0) kullaniliyor. Model,
        # onceki adimda (orn. atalet eksenleri normalizasyonu) zaten
        # merkeze tasindigi icin bu dogru referans noktasidir.

        trsf = gp_Trsf()
        trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale_factor)

        builder = BRepBuilderAPI_Transform(shape, trsf, True)
        olceklenmis_shape = builder.Shape()

        # Dogrulama (sanity check): olcekleme sonrasi bounding box gercekten
        # hedef sinir icinde mi?
        xmin2, xmax2,ymin2, ymax2, zmin2, zmax2 = _bounding_box_al(olceklenmis_shape)
        yeni_maks = max(abs(xmin2), abs(xmax2), abs(ymin2), abs(ymax2), abs(zmin2), abs(zmax2))

        logger.info(
            f"{fileName}: Olcekleme tamamlandi. Yeni bounding box maks_mutlak_deger="
            f"{yeni_maks:.6f} (hedef: {hedef_sinir})"
        )

        if yeni_maks > hedef_sinir + 1e-6:
            logger.warning(
                f"{fileName}: Olcekleme sonrasi bounding box hedef siniri "
                f"({hedef_sinir}) hafifce asiyor: {yeni_maks:.6f}. "
                f"Bu, gp_Trsf/tesselasyon kaynakli kucuk float sapmalari olabilir."
            )

        return olceklenmis_shape
    
    except Exception:
        logger.exception(f"Olcek normalizasyonunda hata: {fileName}")
        return shape