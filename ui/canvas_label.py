from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
class CanvasLabel(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.mouse_last_x = None
        self.mouse_last_y = None

    def mouseMoveEvent(self, event):
        pos = event.position()

        if self.mouse_last_x is None:
            self.mouse_last_x = pos.x()
            self.mouse_last_y = pos.y()
            return

        canvas = self.pixmap()

        painter = QtGui.QPainter(canvas)

        pen = QtGui.QPen()
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setWidth(10)
        painter.setPen(pen)

        painter.drawLine(
            self.mouse_last_x,
            self.mouse_last_y,
            pos.x(),
            pos.y()
        )

        painter.end()
        self.setPixmap(canvas)

        self.mouse_last_x = pos.x()
        self.mouse_last_y = pos.y()

    def mouseReleaseEvent(self, event):
        self.mouse_last_x = None
        self.mouse_last_y = None