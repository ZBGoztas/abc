'''
MODELIN KUTLE DAGILIMINA GORE EN UZUN ORTA KISA EKSENLERINI BULUP Z EKSENINE HIZALAR
'''
from OCC.Core.GProp import GProp_GProps, GProp_PrincipalProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import TopoDS_Shape

from src.braketLog.logger import get_logger
logger = get_logger(__name__)


def _bounding_box_asimetri_kontrolu(shape: TopoDS_Shape) -> tuple:
    """
    Modelin, kendi eksen sistemine gore (0,0,0) merkezinde) her eksende
    pozitif/negatif tarafta ne kadar 'agirlik' (bounding box uzunlugu)
    oldugunu dondurur. Eksen isaretini sabitlemek icin kullanilir.
    """
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)


def atalet_eksenlerine_gore_normalizasyon(fileName: str, shape: TopoDS_Shape) -> TopoDS_Shape:
    """
    Modeli, kutle dagilimina (asal atalet eksenlerine) gore
    (0,0,0) merkezine ve X, Y, Z eksenlerine hizalar.

    - En kucuk atalet momentine sahip eksen -> en 'uzun' eksen -> X'e hizalanir
    - En buyuk atalet momentine sahip eksen -> en 'kisa' eksen -> Z'ye hizalanir
    - Eksenlerin isaret (sign) belirsizligi, bounding box asimetrisine
      bakilarak sabitlenir (ayni parca her zaman ayni yonde hizalanir).

    Not: OCC'nin FirstAxisOfInertia/SecondAxisOfInertia/ThirdAxisOfInertia
    isimlendirmesi, moment buyuklugune gore SIRALI OLMAYABILIR. Bu yuzden
    eksen secimini isimlere gore degil, gercek Moments() degerlerine gore
    yapiyoruz.

    Hata durumunda orijinal shape geri dondurulur (None dondurulmez),
    boylece cagiran kod her zaman gecerli bir TopoDS_Shape alir.
    """
    try:
        # --- 1) FIZIKSEL OZELLIKLERI HESAPLA ---
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)

        # HACIM GECERSIZSE (COK INCE KABUK, BOZUK SHAPE VB.) yuzey ozelliklerine dus
        if props.Mass() <= 1e-9:
            logger.warning(f"{fileName}: Hacim ~0, yuzey ozelliklerine (SurfaceProperties) geciliyor")
            props = GProp_GProps()
            brepgprop.SurfaceProperties(shape, props)
            logger.warning(f"Atalet eksenleri hacim yerine yuzey alani ile hesaplandi | {fileName}")

        merkez = props.CentreOfMass()
        principal: GProp_PrincipalProps = props.PrincipalProperties()

        # DEJENERE DURUM KONTROLU: ASAL EKSENLER KARARSIZSA UYAR
        if principal.HasSymmetryAxis() or principal.HasSymmetryPoint():
            logger.warning(f"{fileName}: Model simetrik/dejenere kutle dagilimina sahip, "
                            f"asal eksen yonelimi kararsiz olabilir")

        I1, I2, I3 = principal.FirstAxisOfInertia(), principal.SecondAxisOfInertia(), principal.ThirdAxisOfInertia()

        # --- GERCEK MOMENT DEGERLERINI AL ---
        # OCC'de First/Second/Third isimlendirmesi moment buyuklugune gore
        # her zaman sirali olmayabilir (test sonuclarinda gozlemlendi).
        # Bu yuzden ekseni ismine gore degil, gercek moment degerine gore secmeliyiz.
        J1, J2, J3 = principal.Moments()

        logger.info(f"1. Atalet Ekseni: ({I1.X():.3f}, {I1.Y():.3f}, {I1.Z():.3f}) | Moment: {J1:.3f}")
        logger.info(f"2. Atalet Ekseni: ({I2.X():.3f}, {I2.Y():.3f}, {I2.Z():.3f}) | Moment: {J2:.3f}")
        logger.info(f"3. Atalet Ekseni: ({I3.X():.3f}, {I3.Y():.3f}, {I3.Z():.3f}) | Moment: {J3:.3f}")

        eksen_moment_listesi = [(J1, I1), (J2, I2), (J3, I3)]
        eksen_moment_listesi.sort(key=lambda item: item[0])  # kucukten buyuge

        # En kucuk moment -> en uzun eksen -> X'e gidecek
        # En buyuk moment -> en kisa eksen -> Z'ye gidecek
        en_uzun_eksen = eksen_moment_listesi[0][1]
        en_kisa_eksen = eksen_moment_listesi[-1][1]

        logger.info(f"En uzun eksen (X'e hizalanacak): "
                    f"({en_uzun_eksen.X():.3f}, {en_uzun_eksen.Y():.3f}, {en_uzun_eksen.Z():.3f})")
        logger.info(f"En kisa eksen (Z'ye hizalanacak): "
                    f"({en_kisa_eksen.X():.3f}, {en_kisa_eksen.Y():.3f}, {en_kisa_eksen.Z():.3f})")

        # --- 2) ISARET (SIGN) BELIRSIZLIGINI SABITLE ---
        # Secilen eksenler +/- yon belirsizligi tasir. Once ham (isaretsiz)
        # eksenlerle deneme donusumu uygulayip, olusan bounding box'un
        # her eksende pozitif tarafta mi negatif tarafta mi daha 'agir' oldugunu
        # kontrol ederek isaretleri sabit bir kurala baglariz:
        # "Her eksende kutlenin/geometrinin cogu +yon'de olsun."

        mevcut_eksen_deneme = gp_Ax3(merkez, gp_Dir(en_kisa_eksen), gp_Dir(en_uzun_eksen))
        trsf_deneme = gp_Trsf()
        trsf_deneme.SetDisplacement(mevcut_eksen_deneme, gp_Ax3())
        deneme_shape = BRepBuilderAPI_Transform(shape, trsf_deneme, True).Shape()

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = _bounding_box_asimetri_kontrolu(deneme_shape)

        # Eksen basina: negatif taraf pozitif taraftan büyükse (mutlak deger) o ekseni ters cevir
        x_flip = abs(xmin) > abs(xmax)
        y_flip = abs(ymin) > abs(ymax)
        z_flip = abs(zmin) > abs(zmax)

        eksen_x_fixed = en_uzun_eksen.Reversed() if x_flip else en_uzun_eksen
        eksen_z_fixed = en_kisa_eksen.Reversed() if z_flip else en_kisa_eksen

        # --- 3) HEDEF VE MEVCUT EKSEN SISTEMLERINI KUR ---
        hedef_eksen = gp_Ax3()  # X=(1,0,0) Y=(0,1,0) Z=(0,0,1), orijin (0,0,0)

        mevcut_eksen = gp_Ax3(
            merkez,
            gp_Dir(eksen_z_fixed),  # Z (en kisa/en buyuk moment)
            gp_Dir(eksen_x_fixed)   # X (en uzun/en kucuk moment)
        )

        # Not: gp_Ax3, Z ve X'ten Y'yi otomatik olarak sag-el kuralina gore hesaplar,
        # boylece secilen eksenler tam ortogonal olmasa da (float hatasi) saglikli
        # bir ortonormal sistem garanti edilir. y_flip degiskeni burada bilgi
        # amaclidir; sag-el kurali bozulmamasi icin sadece X ve Z isaretleri degistirilir.

        # --- 4) DONUSUM MATRISINI HESAPLA VE UYGULA ---
        trsf = gp_Trsf()
        trsf.SetDisplacement(mevcut_eksen, hedef_eksen)
        builder = BRepBuilderAPI_Transform(shape, trsf, True)
        normalize_edilmis_shape = builder.Shape()

        logger.info(f"{fileName}: Atalet eksenlerine gore normalizasyon tamamlandi "
                    f"(x_flip={x_flip}, z_flip={z_flip})")
        return normalize_edilmis_shape
    except Exception:
        logger.exception(f"Atalet eksenlerine gore normalizasyonda hata: {fileName}")
        # Hata durumunda orijinal shape'i dondur, None dondurme (cagiran kod kirilmasin)
        return shape
