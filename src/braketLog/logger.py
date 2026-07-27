import logging
import os
from logging.handlers import RotatingFileHandler
from src.config import log_config  


# ============================================================
# ColorFormatter — Konsola renkli çıktı verir
# ============================================================
class ColorFormatter(logging.Formatter):

    COLORS = {
        "DEBUG"    : "\033[94m",   # 🔵 Mavi    
        "INFO"     : "\033[92m",   # 🟢 Yeşil
        "WARNING"  : "\033[93m",   # 🟡 Sarı
        "ERROR"    : "\033[91m",   # 🔴 Kırmızı
        "CRITICAL" : "\033[95m",   # 🟣 Mor
    }
    RESET = "\033[0m"              

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ============================================================
# Yardımcı — Seviyeye göre dosya handler oluşturur
# ============================================================
# Sadece tam olarak o seviyeyi geçirir, üstünü/altını geçirmez
class ExactLevelFilter(logging.Filter):
    def __init__(self, level: int):
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level  # Sadece eşitse True döner


def _build_file_handler(log_path: str, level: int) -> logging.Handler:
    if log_config.LOG_ROTATING:
        handler = RotatingFileHandler(
            filename    = log_path,
            maxBytes    = log_config.MAX_BYTES,
            backupCount = log_config.BACKUP_COUNT,
            encoding    = 'utf-8'
        )
    else:
        handler = logging.FileHandler(log_path, encoding='utf-8')

    handler.setLevel(level)
    handler.addFilter(ExactLevelFilter(level))   # ✅ Filtre eklendi!
    handler.setFormatter(
        logging.Formatter(log_config.LOG_FORMAT, datefmt=log_config.DATE_FORMAT)
    )
    return handler


# ============================================================
# get_logger — Her modülden çağrılacak ana fonksiyon
# ============================================================
def get_logger(name: str = __name__) -> logging.Logger:
    """
    Merkezi logger oluşturur ve döndürür.

    Kullanım:
        from src.logging import get_logger
        logger = get_logger(__name__)

    Parametreler:
        name (str): __name__ ile çağrılırsa modül adını otomatik alır.
                    Log satırında hangi dosyadan geldiği görünür.

    Dönüş:
        logging.Logger: Yapılandırılmış logger nesnesi.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_config.LOG_LVL)

    # Duplicate handler koruması
    if logger.handlers:
        return logger

    # ── Log klasörü oluştur ───────────────────────────────────
    os.makedirs(log_config.LOG_DIR, exist_ok=True)

    # ── Seviyeye göre ayrı dosya handler'ları ────────────────
    for level, filename in log_config.LOG_FILE.items():
        log_path = os.path.join(log_config.LOG_DIR, filename)
        file_handler = _build_file_handler(log_path, level)
        logger.addHandler(file_handler)

    # ── Konsol Handler (renkli) ───────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_config.LOG_LVL)
    console_handler.setFormatter(
        ColorFormatter(log_config.LOG_FORMAT, datefmt=log_config.DATE_FORMAT)
    )
    logger.addHandler(console_handler)

    return logger
