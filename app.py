import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from smg.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sprite Soul")
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
