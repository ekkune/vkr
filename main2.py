import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

class ExponentialMatrixApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        matrix = np.array([[1, 2], [3, 4]])
        exp_matrix = np.exp(matrix)

        for row in exp_matrix:
            layout.addWidget(QLabel(str(row)))

        self.setLayout(layout)
        self.setWindowTitle('Exponential of Matrix Coefficients')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExponentialMatrixApp()
    sys.exit(app.exec())