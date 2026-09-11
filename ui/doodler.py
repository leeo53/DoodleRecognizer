import math
import random
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

from ui.canvas_label import CanvasLabel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from network.nn import NeuralNetwork


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Doodle Recognizer")
        self.central_widget = QtWidgets.QWidget()
        self.canvas_label = CanvasLabel()
        canvas = QtGui.QPixmap(560, 560)
        canvas.fill(Qt.white)
        self.canvas_label.setPixmap(canvas)
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.canvas_layout = QtWidgets.QHBoxLayout()

        self.label_stylesheet = """
                QLabel {
                    background-color: white;
                    border-radius: 10px;
                    padding: 6px 10px;
                    font-size: 14px;
                }
            """

        self.header_stylesheet = """
            QLabel {
                font-size: 20px;
                font-weight: bold;
                padding: 6px;
            }
        """

        self.progress_stylesheet = """
        QProgressBar {
            background-color: white;
            border: none;
            border-radius: 10px;
            text-align: center;
            font-size: 14px;
        }

        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 6px;
        }
        """

        self.cnn_guesses_layout = QtWidgets.QVBoxLayout()
        self.cnn_g_label = QtWidgets.QLabel("CNN guesses")
        self.cnn_g_label.setStyleSheet(self.header_stylesheet)
        self.cnn_guesses = QtWidgets.QGridLayout()
        self.cnn_guesses_layout.addWidget(self.cnn_g_label)
        self.cnn_guesses_layout.addLayout(self.cnn_guesses)
        self.cnn_name_labels = []
        self.cnn_prob_bars = []

        self.cnn_model = NeuralNetwork()
        self.cnn_model.load("doodle_cnn.pkl")

        self.simple_nn_guesses_layout = QtWidgets.QVBoxLayout()
        self.simple_nn_g_label = QtWidgets.QLabel("Simple NN guesses")
        self.simple_nn_g_label.setStyleSheet(self.header_stylesheet)
        self.simple_nn_guesses = QtWidgets.QGridLayout()
        self.simple_nn_guesses_layout.addWidget(self.simple_nn_g_label)
        self.simple_nn_guesses_layout.addLayout(self.simple_nn_guesses)
        self.simple_nn_name_labels = []
        self.simple_nn_prob_bars = []
        self.nn_model = NeuralNetwork()
        self.nn_model.load("doodle_simple_nn.pkl")

        self.guesses_layout = QtWidgets.QVBoxLayout()
        self.guesses_layout.addLayout(self.cnn_guesses_layout)
        self.guesses_layout.addLayout(self.simple_nn_guesses_layout)
        self.guesses_layout.addStretch()

        for row in range(5):
            cnn_name_label = QtWidgets.QLabel()
            cnn_name_label.setStyleSheet(self.label_stylesheet)
            cnn_name_label.setFixedHeight(36)
            cnn_probs_bar = QtWidgets.QProgressBar()
            cnn_probs_bar.setRange(0, 1000)
            cnn_probs_bar.setFixedHeight(36)
            cnn_probs_bar.setMinimumWidth(180)
            cnn_probs_bar.setTextVisible(True)
            cnn_probs_bar.setStyleSheet(self.progress_stylesheet)

            nn_name_label = QtWidgets.QLabel()
            nn_name_label.setStyleSheet(self.label_stylesheet)
            nn_name_label.setFixedHeight(36)
            nn_probs_bar = QtWidgets.QProgressBar()
            nn_probs_bar.setRange(0, 1000)
            nn_probs_bar.setFixedHeight(36)
            nn_probs_bar.setMinimumWidth(180)
            nn_probs_bar.setTextVisible(True)
            nn_probs_bar.setStyleSheet(self.progress_stylesheet)

            self.cnn_guesses.addWidget(cnn_name_label, row, 0)
            self.cnn_guesses.addWidget(cnn_probs_bar, row, 1)
            self.simple_nn_guesses.addWidget(nn_name_label, row, 0)
            self.simple_nn_guesses.addWidget(nn_probs_bar, row, 1)

            self.cnn_name_labels.append(cnn_name_label)
            self.cnn_prob_bars.append(cnn_probs_bar)
            self.simple_nn_name_labels.append(nn_name_label)
            self.simple_nn_prob_bars.append(nn_probs_bar)

        self.guess_button = QtWidgets.QPushButton("Guess")
        self.guess_button.clicked.connect(self.guess_pressed)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_pressed)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.guess_button)
        self.button_layout.addWidget(self.clear_button)

        self.canvas_layout.addWidget(self.canvas_label)
        self.canvas_layout.addLayout(self.guesses_layout)
        self.layout.addLayout(self.canvas_layout)
        self.layout.addLayout(self.button_layout)


    def guess_pressed(self):
        '''
        sends the doodle to the models and shows the top 5 predictions with probabilities in the
        guesses section of the ui
        '''
        image = self.canvas_label.pixmap().toImage()

        image = image.scaled(
            28,
            28,
            Qt.IgnoreAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        image = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)

        array = np.frombuffer(
            image.bits(),
            dtype=np.uint8
        ).reshape(image.height(),image.bytesPerLine())

        array = array[:,:image.width()].copy()

        tensor = torch.from_numpy(array).float() / 255.0
        tensor = tensor.to(device)
        tensor = (1.0 - tensor) ** .2

        cnn_input = tensor.reshape(1,1,28,28)
        nn_input = tensor.reshape(1,-1)
        cnn_predicted_names, cnn_probs = self.cnn_model.predict(cnn_input)
        nn_predicted_names, nn_probs = self.nn_model.predict(nn_input)
        cnn_names = cnn_predicted_names[0]
        cnn_probs = cnn_probs[0]
        nn_names = nn_predicted_names[0]
        nn_probs = nn_probs[0]

        for i, (name, prob) in enumerate(zip(cnn_names, cnn_probs)):
            self.cnn_name_labels[i].setText(Path(name).stem)
            self.cnn_prob_bars[i].setValue(round(prob.item() * 1000))
            self.cnn_prob_bars[i].setFormat(f"{prob.item() * 100:.1f}%")
            if i == 0:
                self.cnn_name_labels[i].setStyleSheet("""
                        QLabel {
                            background-color: white;
                            border-radius: 10px;
                            padding: 8px 12px;
                            font-size: 20px;
                            font-weight: bold;
                        }
                    """)
        for i, (name, prob) in enumerate(zip(nn_names, nn_probs)):
            self.simple_nn_name_labels[i].setText(Path(name).stem)
            self.simple_nn_prob_bars[i].setValue(round(prob.item() * 1000))
            self.simple_nn_prob_bars[i].setFormat(f"{prob.item() * 100:.1f}%")
            if i == 0:
                self.simple_nn_name_labels[i].setStyleSheet("""
                        QLabel {
                            background-color: white;
                            border-radius: 10px;
                            padding: 8px 12px;
                            font-size: 20px;
                            font-weight: bold;
                        }
                    """)

    def clear_pressed(self):
        '''
        clears the canvas and the guesses
        '''
        canvas = QtGui.QPixmap(560, 560)
        canvas.fill(Qt.white)
        self.canvas_label.setPixmap(canvas)

        for name_label, prob_bar in zip(
                self.cnn_name_labels,
                self.cnn_prob_bars
        ):
            name_label.clear()
            prob_bar.setValue(0)
            prob_bar.setFormat("")

        for name_label, prob_bar in zip(
                self.simple_nn_name_labels,
                self.simple_nn_prob_bars
        ):
            name_label.clear()
            prob_bar.setValue(0)
            prob_bar.setFormat("")


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
window.setFixedSize(window.size())
app.exec()