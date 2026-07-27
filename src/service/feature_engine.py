from OCC.Core.STEPControl import STEPControl_Reader  # Step formatindaki cad dosyalarini okumak ve sisteme aktarmak icin kullanilir
from OCC.Core.IFSelect import IFSelect_RetDone  # Bir okuma/aktarma isleminin basari ile tamamlandigini kontrol etmek icin kullanilan durum sabitidir

from OCC.Core.TopExp import topexp
# TopExp: MapShapes() gibi statik metodlarla bir sekildeki (Shape) benzersiz alt ogeleri (yuzey, kenar, kose) indeksli sekilde toplamak icin kullanilir
# topexp: MapShapesAndAncestors() ile bir alt ogenin (orn. kenar) hangi ust ogelere (orn. yuzey) ait oldugunu (komsuluk/adjacency) bulmak icin kullanilir

from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_REVERSED
# TopAbs_FACE / TopAbs_EDGE / TopAbs_VERTEX: Topolojik oge turlerini (yuzey, kenar, kose) belirten sabitlerdir, gezinme/filtreleme yaparken kullanilir
# TopAbs_REVERSED: Bir yuzeyin/kenarin ters yonlu (orijinal geometriye gore ters cevrilmis) oldugunu belirten yonelim sabitidir, normal vektorunun yonunu duzeltmek icin kullanilir

from OCC.Core.TopoDS import topods  # Genel sekilleri (TopoDS_Shape), daha spesifik topolojik tiplere (Face, Edge, Vertex) donusturmek icin kullanilir

'''
Geometrik yuzey tiplerini temsil eden sabitlerdir (Duzlem, Silindir, BSpline Yuzeyi, Bezier Yuzeyi, Ekstruzyon Yuzeyi vb.)
Bir yuzeyin (face) matematiksel olarak hangi tur geometriye ait oldugunu tespit etmek icin BRepAdaptor_Surface.GetType() ile birlikte kullanilir
'''
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_Cone, GeomAbs_OffsetSurface, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_SurfaceOfRevolution
)

from OCC.Core.GProp import GProp_GProps  # Bir modelin kutle, hacim, yuzey alani, agirlik merkezi ve eylemsizlik momenti gibi fiziksel ozelliklerini tutan siniftir
from OCC.Core.BRepGProp import brepgprop  # Topolojik sekillerin (BRep) hacim, alan veya kutle gibi ozelliklerini hesaplayip GProp_GProps nesnesine yukleyen hesaplama araclarini icerir

from OCC.Core.Bnd import Bnd_Box  # 3 Boyutlu uzayda bir sinirlayici kutu (bounding box) olusturur, modelin minimum ve maksimum X, Y, Z koordinat sinirlarini tutar
from OCC.Core.BRepBndLib import brepbndlib  # Bir topolojik sekli tamamen icine alan sinirlayici kutunun (Bnd_Box) sinirlarini hesaplamak icin kullanilir

from OCC.Core.BRep import BRep_Tool  # Bir yuzeyin (Face) altinda yatan gercek geometrik yuzey tanimini (Geom_Surface) elde etmek gibi dusuk seviyeli BRep sorgulari icin kullanilir

from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,  # Bir alt ogeye (orn. kenar) bagli birden fazla ust ogeyi (orn. yuzey listesi) indeksli sekilde tutan komsuluk haritasi veri yapisidir
    TopTools_IndexedMapOfShape,                 # Bir sekildeki alt ogeleri (yuzey/kenar/kose) TEKRARSIZ ve indeksli olarak saklayan veri yapisidir (benzersizligi garanti eder)
    TopTools_ListIteratorOfListOfShape          # TopTools_IndexedDataMapOfShapeListOfShape icinden donen sekil listelerini (orn. bir kenara bagli yuzeyler) gezmek icin kullanilan iterator sinifidir
)

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # Bir yuzeyin (Face) geometrik tipini (Plane, Cylinder, Sphere vb.) ve U/V parametre sinirlarini sorgulamak icin kullanilan adaptor sinifidir

from OCC.Core.GeomLProp import GeomLProp_SLProps  # Bir yuzey uzerindeki belirli bir (u, v) noktasinda normal vektoru, egrilik gibi yerel (local) diferansiyel geometrik ozellikleri hesaplamak icin kullanilir

from src.braketLog.logger import get_logger  # Proje icin ozellestirilmis, dosya/konsol loglama yapmayi saglayan logger olusturucu fonksiyondur
import os  # Dosya yolu islemleri (orn. dosya adini yoldan ayirmak) icin kullanilan standart kutuphanedir



class FeatureEngine():

    def __init__(self, filePath):
        self.logger = get_logger(__name__)
        self.filePath = filePath
        self.baseName = os.path.basename(filePath)
        self.shape = self.stepReader()
        if self.shape is None:
            raise ValueError(f"Step dosyasi yuklenemedi: {self.baseName}")

        # --- CACHE ALANLARI (TEKRAR HESAPLAMAYI ONLEMEK ICIN) ---
        self._bbox_cache = None
        self._physical_cache = None
        self._topoloji_cache = None
        self._yuzey_tipleri_cache = None
        self._yuzey_normalleri_cache = None

    def stepReader(self):
        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(self.filePath)

            if status != IFSelect_RetDone:
                raise Exception(f"Step dosyasi okunamadi: {status}")
            reader.TransferRoots()
            shape = reader.OneShape()

            self.logger.info(f"Step dosyasi basari ile yuklendi: {self.baseName}")
            return shape

        except Exception:
            self.logger.exception(f"Step dosyasi okunurken hata olustu: {self.baseName}")
            return None

    def bounding_box(self):
        # --- CACHE KONTROLU ---
        if self._bbox_cache is not None:
            return self._bbox_cache

        try:
            bbox = Bnd_Box()
            brepbndlib.Add(self.shape, bbox)

            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            boyutlar = {
                "x": xmax - xmin,
                "y": ymax - ymin,
                "z": zmax - zmin,
                "merkez": (
                    (xmax + xmin) / 2,
                    (ymax + ymin) / 2,
                    (zmax + zmin) / 2
                )
            }
            self.logger.info(f"Bounding Box hesaplandi: {self.baseName}")

            # --- CACHE'E YAZ ---
            self._bbox_cache = boyutlar
            return boyutlar

        except Exception:
            self.logger.exception(f"Bounding Box olusturulurken hata olustu: {self.baseName}")
            return None

    def physical_features(self):
        # --- CACHE KONTROLU ---
        if self._physical_cache is not None:
            return self._physical_cache

        try:
            props = GProp_GProps()
            brepgprop.VolumeProperties(self.shape, props)

            volume = props.Mass()
            agirlik_merkezi = props.CentreOfMass()

            self.logger.info(f"Hacim hesaplandi: {self.baseName}-{volume}")
            self.logger.info(f"Agirlik merkezi hesaplandi: {self.baseName}-{agirlik_merkezi}")

            physical_features = {
                "hacim": volume,
                "agirlikMerkezi": agirlik_merkezi
            }

            # --- CACHE'E YAZ ---
            self._physical_cache = physical_features
            return physical_features

        except Exception:
            self.logger.exception(f"Fiziksel ozellikler hesaplanirken hata olustu: {self.baseName}")
            return None

    def topoloji_gez(self):
        if self._topoloji_cache is not None:
            return self._topoloji_cache

        try:
            # --- 1. BENZERSIZ SEKILLER ---
            face_map = TopTools_IndexedMapOfShape()
            topexp.MapShapes(self.shape, TopAbs_FACE, face_map)
            # OCC indexed map'leri 1-based calisir; bu binding'de sayi bilgisi Size() ile alinmali.
            yuzeyler = [topods.Face(face_map.FindKey(i)) for i in range(1, face_map.Size() + 1)]

            edge_map = TopTools_IndexedMapOfShape()
            topexp.MapShapes(self.shape, TopAbs_EDGE, edge_map)
            kenarlar = [topods.Edge(edge_map.FindKey(i)) for i in range(1, edge_map.Size() + 1)]

            vertex_map = TopTools_IndexedMapOfShape()
            topexp.MapShapes(self.shape, TopAbs_VERTEX, vertex_map)
            noktalar = [topods.Vertex(vertex_map.FindKey(i)) for i in range(1, vertex_map.Size() + 1)]

            # --- 2. EDGE -> FACE KOMSULUGU ---
            edge_to_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(self.shape, TopAbs_EDGE, TopAbs_FACE, edge_to_face_map)

            kenar_yuzey_baglantilari = {}

            # Edge->face eslesmesi de ayni 1-based indeks mantigini takip eder.
            for i in range(1, edge_map.Size() + 1):
                edge = edge_map.FindKey(i)
                faces_of_edge = edge_to_face_map.FindFromKey(edge)

                komsu_yuzey_indeksleri = []
                iterator = TopTools_ListIteratorOfListOfShape(faces_of_edge)
                while iterator.More():
                    face = iterator.Value()
                    # FindIndex() 1-based doner; 0 gelirse face map'te bulunamamistir.
                    face_idx = face_map.FindIndex(face)
                    if face_idx <= 0:
                        raise ValueError("Yuzey indexi map icinde bulunamadi")
                    face_idx -= 1
                    komsu_yuzey_indeksleri.append(face_idx)
                    iterator.Next()
                kenar_yuzey_baglantilari[i - 1] = komsu_yuzey_indeksleri

            # --- 3. FACE -> FACE KOMSULUGU (GNN icin en kritik yapi) ---
            face_komsuluk = {i: set() for i in range(len(yuzeyler))}
            for edge_idx, komsu_yuzeyler in kenar_yuzey_baglantilari.items():
                if len(komsu_yuzeyler) == 2:
                    f1, f2 = komsu_yuzeyler
                    face_komsuluk[f1].add(f2)
                    face_komsuluk[f2].add(f1)
            face_komsuluk = {k: list(v) for k, v in face_komsuluk.items()}

            topoloji_features = {
                "nodes": noktalar,
                "edges": kenarlar,
                "faces": yuzeyler,
                "adjacency_edge_to_face": kenar_yuzey_baglantilari,
                "adjacency_face_to_face": face_komsuluk
            }

            self.logger.info(f"Topoljik ozellikler cikarildi: {os.path.basename(self.filePath)}")
            self._topoloji_cache = topoloji_features
            return topoloji_features

        except Exception:
            self.logger.exception(f"Topoloji gezilirken hata olustu: {self.baseName}")
            return None

    def yuzey_tipleri_tespit_et(self, shapes=None):
        """
        shapes verilmezse, topoloji_gez() cache'inden yuzeyleri otomatik alir.
        Boylece disaridan tekrar face listesi gecirmeye gerek kalmaz.
        """
        # --- CACHE KONTROLU ---
        if self._yuzey_tipleri_cache is not None:
            return self._yuzey_tipleri_cache

        if shapes is None:
            shapes = self.topoloji_gez()['faces']

        yuzeyler = {
            "duzYuzeyler": [],
            "silindirikYuzeyler": [],
            "koniYuzeyler": [],
            "kureYuzeyler": [],
            "torusYuzeyler": [],
            "BSplineYuzeyler": [],
            "BezierYuzeyler": [],
            "devirYuzeyler": [],
            "SurfaceOfExtrusion": [],
            "offsetYuzeyler": [],
            "digerYuzeyler": []
        }
        try:
            for face in shapes:
                adaptor = BRepAdaptor_Surface(face, True)
                tip = adaptor.GetType()

                if tip == GeomAbs_Plane:
                    self.logger.info("Düzlem (Plane) yüzeyi bulundu")
                    yuzeyler['duzYuzeyler'].append(face)
                elif tip == GeomAbs_Cylinder:
                    self.logger.info("Silindir yüzeyi bulundu")
                    yuzeyler['silindirikYuzeyler'].append(face)
                elif tip == GeomAbs_Cone:
                    self.logger.info("Koni yüzeyi bulundu")
                    yuzeyler['koniYuzeyler'].append(face)
                elif tip == GeomAbs_Sphere:
                    self.logger.info("Küre yüzeyi bulundu")
                    yuzeyler['kureYuzeyler'].append(face)
                elif tip == GeomAbs_Torus:
                    self.logger.info("Torus yüzeyi bulundu")
                    yuzeyler['torusYuzeyler'].append(face)
                elif tip == GeomAbs_BSplineSurface:
                    self.logger.info("BSpline yüzeyi bulundu")
                    yuzeyler['BSplineYuzeyler'].append(face)
                elif tip == GeomAbs_BezierSurface:
                    self.logger.info("BezierSurface yüzeyi bulundu")
                    yuzeyler['BezierYuzeyler'].append(face)
                elif tip == GeomAbs_SurfaceOfRevolution:
                    self.logger.info("Devir yüzeyi bulundu")
                    yuzeyler['devirYuzeyler'].append(face)
                elif tip == GeomAbs_SurfaceOfExtrusion:
                    self.logger.info("SurfaceOfExtrusion yüzeyi bulundu")
                    yuzeyler['SurfaceOfExtrusion'].append(face)
                elif tip == GeomAbs_OffsetSurface:
                    self.logger.info("Offset yüzeyi bulundu")
                    yuzeyler['offsetYuzeyler'].append(face)
                else:
                    self.logger.info("Diger/Bilinmeyen yuzey tipi bulundu")
                    yuzeyler['digerYuzeyler'].append(face)

            self._yuzey_tipleri_cache = yuzeyler
            return yuzeyler
        except Exception:
            self.logger.exception(f"Yuzey tipleri tespit edilirken hata olustu {os.path.basename(self.filePath)}")
            return None

    def _tek_normal_hesapla(self, face, surface_geom, u, v):
        """
        Yardimci fonksiyon: verilen (u, v) noktasinda normal vektorunu hesaplar.
        Disaridan cagirilmasi beklenmez (private).
        """
        props = GeomLProp_SLProps(surface_geom, u, v, 1, 1e-6)

        if not props.IsNormalDefined():
            return None

        normal = props.Normal()

        if face.Orientation() == TopAbs_REVERSED:
            normal.Reverse()

        return (normal.X(), normal.Y(), normal.Z())

    def _hesapla_tek_yuzey(self, face, is_planar, grid_size):
        """
        Tek bir yuzey icin normal hesaplama mantigini yurutur.
        is_planar=True ise tek nokta, degilse grid_size x grid_size grid kullanir.
        """
        surface_geom = BRep_Tool.Surface(face)
        adaptor = BRepAdaptor_Surface(face, True)

        u_min = adaptor.FirstUParameter()
        u_max = adaptor.LastUParameter()
        v_min = adaptor.FirstVParameter()
        v_max = adaptor.LastVParameter()

        normaller = []

        if is_planar:
            u_mid = (u_min + u_max) / 2.0
            v_mid = (v_min + v_max) / 2.0

            n = self._tek_normal_hesapla(face, surface_geom, u_mid, v_mid)

            if n is not None:
                normaller.append(n)

        else:
            for i in range(grid_size):
                for j in range(grid_size):
                    u = u_min + (u_max - u_min) * (i + 0.5) / grid_size
                    v = v_min + (v_max - v_min) * (j + 0.5) / grid_size
                    n = self._tek_normal_hesapla(face, surface_geom, u, v)
                    if n is not None:
                        normaller.append(n)
        if not normaller:
            return None
        return {
            "face": face,
            "is_planer": is_planar,
            "normals": normaller
        }

    def yuzey_normalleri(self, grid_size=5):
        """
        Her yuzey icin adaptif normal hesaplama yapar:
          - Plane ise: tek merkez nokta normali (tek hesap, tekrar yok)
          - Kivrimli ise: grid_size x grid_size UV izgarasi ile normal listesi

        Sonuc listesi, topoloji_gez()['faces'] ile AYNI SIRADA doner.
        Bu, BRepNet gibi GNN mimarileri icin index tutarliligi saglar.
        """

        # --- CACHE KONTROLU ---
        if self._yuzey_normalleri_cache is not None:
            return self._yuzey_normalleri_cache

        try:
            # 1. Yuzey tiplerini al (Cache'den gelir, hizlidir)
            yuzey_tipleri_dict = self.yuzey_tipleri_tespit_et()

            # NOT: set() burada nesne kimligine (id) dayanir; topoloji_gez()
            # cache'inden gelen ayni face referanslari kullanildigi surece
            # bu karsilastirma guvenlidir. Disaridan farkli bir face listesi
            # verilirse (yuzey_tipleri_tespit_et(shapes=...) ile) bu varsayim
            # kirilabilir.
            duz_yuzeyler_seti = set(yuzey_tipleri_dict["duzYuzeyler"])

            sonuclar = []

            # 2. Yuzeyleri Orijinal Topoloji Sirasina Gore Gez!
            orijinal_yuzey_sirasi = self.topoloji_gez()['faces']

            for face in orijinal_yuzey_sirasi:
                is_planar = face in duz_yuzeyler_seti
                sonuc = self._hesapla_tek_yuzey(face, is_planar=is_planar, grid_size=grid_size)
                sonuclar.append(sonuc)  # None gelirse de sira bozulmasin diye eklenir

            self.logger.info(f"Yuzey normalleri hesaplandi: {os.path.basename(self.filePath)}")
            self._yuzey_normalleri_cache = sonuclar
            return sonuclar

        except Exception:
            self.logger.exception(f"Yuzey normalleri hesaplanirken hata olustu {self.baseName}")
            return None

    def tum_ozellikleri_getir(self, grid_size=5):
        """
        Parca hakkinda tum feature'lari tek bir sozlukte toplar.
        Ic ice bagimli metodlari (topoloji -> yuzey tipleri -> normaller)
        dogru sirada tetikler; cache sayesinde tekrar hesaplama yapilmaz.
        """
        try:
            rapor = {
                "dosyaAdi": self.baseName,
                "boundingBox": self.bounding_box(),
                "fizikselOzellikler": self.physical_features(),
                "topoloji": self.topoloji_gez(),
                "yuzeyTipleri": self.yuzey_tipleri_tespit_et(),
                "yuzeyNormalleri": self.yuzey_normalleri(grid_size=grid_size),
            }
            self.logger.info(f"Tum ozellikler basari ile toplandi: {self.baseName}")
            return rapor
        except Exception:
            self.logger.exception(f"Ozellikler toplanirken hata olustu: {self.baseName}")
            return None
