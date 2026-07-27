from pathlib import Path
import random
import shutil
import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

# Backend'i qtViewer3d import edilmeden önce yükle
from OCC.Display.backend import load_backend
for _backend in ("pyqt5", "pyside2", "pyqt6"):
    try:
        load_backend(_backend)
        break
    except Exception:
        pass

from OCC.Display.qtDisplay import qtViewer3d

try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    from PySide2 import QtWidgets, QtCore


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data" / "abc" / "raw" / "step_data"
DATASET_DIR = BASE_DIR / "data" / "abc" / "dataset"
REJECTED_DIR = BASE_DIR / "data" / "abc" / "rejected"

STEP_FILES = sorted(
    list(SOURCE_DIR.glob("*.stp")) + list(SOURCE_DIR.glob("*.step"))
)
random.shuffle(STEP_FILES)


def count_step_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.stp"))) + len(list(folder.glob("*.step")))


def load_step_file(path: Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise FileNotFoundError(f"Okunamadı: {path}")
    reader.TransferRoots()
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for i in range(1, reader.NbShapes() + 1):
        builder.Add(compound, reader.Shape(i))
    return compound


def move_file(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"Zaten var: {dst}")
    shutil.move(str(src), str(dst))


class DatasetLabeler(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.idx = 0
        self.session_added_count = 0
        self.session_removed_count = 0
        self.setWindowTitle("STEP Dataset Kontrol")
        self.resize(1280, 720)

        # ── Ana layout ────────────────────────────────────────────────────
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 3D görüntüleyici ──────────────────────────────────────────────
        self.canvas = qtViewer3d(self)
        root_layout.addWidget(self.canvas, stretch=5)

        # ── Sağ panel ─────────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(220)
        panel.setStyleSheet("background:#1e1e1e;")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 16, 12, 16)
        panel_layout.setSpacing(10)

        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.setStyleSheet("color:#aaa; font-size:11px;")
        panel_layout.addWidget(self.progress_label)

        self.file_label = QtWidgets.QLabel()
        self.file_label.setWordWrap(True)
        self.file_label.setAlignment(QtCore.Qt.AlignCenter)
        self.file_label.setStyleSheet(
            "color:#fff; font-size:12px; padding:8px;"
        )
        panel_layout.addWidget(self.file_label)

        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stats_label.setStyleSheet("color:#cfcfcf; font-size:11px; padding:6px;")
        panel_layout.addWidget(self.stats_label)

        panel_layout.addStretch()

        for text, color, slot in [
            ("✔  Dataset'e Ekle",      "#4CAF50", self.add_to_dataset),
            ("✘  Dataset'ten Çıkart",  "#f44336", self.remove_from_dataset),
            ("→  Geç (Sonraki)",       "#2196F3", self.skip_current),
        ]:
            panel_layout.addWidget(self._make_button(text, color, slot))

        root_layout.addWidget(panel)

        self.occ = None
        self._occ_ready = False
        self._refresh()
        self._update_stats_label()

    # ── OCC widget göründükten sonra başlat ───────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not self._occ_ready:
            self._occ_ready = True
            self.canvas.InitDriver()
            self.occ = self.canvas._display
            self.occ.set_bg_gradient_color([40, 40, 40], [80, 80, 80])
            QtCore.QTimer.singleShot(100, self._show_current)

    # ── Yardımcılar ───────────────────────────────────────────────────────

    @staticmethod
    def _make_button(text, color, slot):
        btn = QtWidgets.QPushButton(text)
        btn.setFixedHeight(44)
        btn.setStyleSheet(
            f"QPushButton {{ background:{color}; color:white; font-size:13px;"
            f" border-radius:6px; }}"
            f"QPushButton:hover {{ background:{color}bb; }}"
        )
        btn.clicked.connect(slot)
        return btn

    def _refresh(self):
        total = len(STEP_FILES)
        if self.idx < total:
            self.progress_label.setText(f"{self.idx + 1} / {total}")
            self.file_label.setText(STEP_FILES[self.idx].name)
        else:
            self.progress_label.setText("")
            self.file_label.setText("Tüm dosyalar işlendi")

    def _update_stats_label(self):
        dataset_total = count_step_files(DATASET_DIR)
        rejected_total = count_step_files(REJECTED_DIR)
        self.stats_label.setText(
            "\n".join(
                [   
                    f"Toplam Islenen Dosya {dataset_total + rejected_total}",
                    f"Dataset toplam: {dataset_total}",
                    f"Cikarilan toplam: {rejected_total}",
                    f"Bu oturum eklenen: {self.session_added_count}",
                    f"Bu oturum cikarilan: {self.session_removed_count}",
                ]
            )
        )

    def _show_current(self):
        if self.occ is None or self.idx >= len(STEP_FILES):
            return
        shape = load_step_file(STEP_FILES[self.idx])
        self.occ.EraseAll()
        self.occ.DisplayShape(shape, update=True, color="BLUE1", transparency=0.0)
        self.occ.FitAll()

    def _advance(self):
        self.idx += 1
        if self.idx >= len(STEP_FILES):
            QtWidgets.QMessageBox.information(
                self,
                "Bitti",
                "\n".join(
                    [
                        "Tum dosyalar islendi.",
                        f"Bu oturum dataset'e eklenen: {self.session_added_count}",
                        f"Bu oturum dataset'ten cikarilan: {self.session_removed_count}",
                    ]
                ),
            )
            self._refresh()
            self._update_stats_label()
            return
        self._refresh()
        self._show_current()

    # ── Buton aksiyonları ─────────────────────────────────────────────────

    def add_to_dataset(self):
        try:
            move_file(STEP_FILES[self.idx], DATASET_DIR)
            self.session_added_count += 1
            self._update_stats_label()
            self._advance()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def remove_from_dataset(self):
        try:
            move_file(STEP_FILES[self.idx], REJECTED_DIR)
            self.session_removed_count += 1
            self._update_stats_label()
            self._advance()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def skip_current(self):
        self._advance()


if __name__ == "__main__":
    if not STEP_FILES:
        raise FileNotFoundError(f"{SOURCE_DIR} içinde STEP dosyası bulunamadı.")

    app = QtWidgets.QApplication(sys.argv)
    window = DatasetLabeler()
    window.show()
    sys.exit(app.exec_())
