from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
import os

class ModelAccuracyPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        self.setWindowTitle("Model Accuracy")
        self.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout()

        self.confusion_label = QLabel()
        confusion_image_path = "my_confusion_matrix.png"
        if os.path.exists(confusion_image_path):
            pixmap = QPixmap(confusion_image_path)
            self.confusion_label.setPixmap(pixmap)
        else:
            self.confusion_label.setText("Confusion matrix image not found.")
        layout.addWidget(self.confusion_label)

        # Button to go back
        self.back_button = QPushButton("Back to Main Page")
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def go_back(self):
        """Go back to the main page."""
        self.controller.show_main_page()
        self.confusion_label.hide()

