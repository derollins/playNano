"""Main entry point for the playNano GUI application."""

import sys
from PySide6.QtWidgets import QApplication
from playNano.gui.window import MainWindow


def gui_entry(args):
    """Entry point for the playNano GUI application."""
    app = QApplication(sys.argv)
    wnd = MainWindow(args.input_file)
    wnd.show()
    sys.exit(app.exec())
