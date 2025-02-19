from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, 
    QTableWidgetItem, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt
from database import create_connection, getname


class MainPage(QWidget):
    def __init__(self, controller, user):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Main Page")
        self.setGeometry(100, 100, 600, 400)
        self.current_user = user

        # --- Top Layout ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logout Button
        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)
        top_layout.addWidget(self.logout_button, alignment=Qt.AlignLeft)

        # Welcome Label
        self.welcome_label = QLabel(f"Welcome, {getname(self.controller.conn ,self.current_user)}") 
        self.welcome_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        top_layout.addWidget(self.welcome_label, alignment=Qt.AlignCenter)

        # Spacer
        top_layout.addStretch()

        # --- Center Layout ---
        center_layout = QVBoxLayout()
        center_layout.setSpacing(0)
        center_layout.setContentsMargins(0, 0, 0, 0)

        center_layout.addStretch()
        
        # Start Video Button
        self.start_video_button = QPushButton("Start Video")
        self.start_video_button.clicked.connect(self.start_video)
        self.start_video_button.setStyleSheet("font-size: 16px;")
        center_layout.addWidget(self.start_video_button, alignment=Qt.AlignCenter)
        
        center_layout.addStretch()

        # --- Bottom Layout ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        # Bottom Left: Table for Model Accuracy
        self.model_accuracy_button = QPushButton("Model Accuracy")
        self.model_accuracy_button.clicked.connect(self.show_model_accuracy_page)
        bottom_layout.addWidget(self.model_accuracy_button, alignment=Qt.AlignLeft)
        
        
        # Bottom Right: History Button
        self.history_button = QPushButton("History")
        self.history_button.clicked.connect(self.show_history)
        bottom_layout.addWidget(self.history_button, alignment=Qt.AlignRight)

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def populate_accuracy_table(self):
        """Populate the model accuracy table with example data."""
        metrics = ["Accuracy", "Precision", "Recall"]
        values = ["95%", "92%", "90%"]  # Replace with real data as needed
        for row, (metric, value) in enumerate(zip(metrics, values)):
            self.accuracy_table.setItem(row, 0, QTableWidgetItem(metric))
            self.accuracy_table.setItem(row, 1, QTableWidgetItem(value))

    def logout(self):
        """Handle logout and navigate to the login page."""
        if self.controller:
            self.controller.set_current_user(None)
            self.controller.show_login_page()

    def start_video(self):
        """Handle start video action."""
        self.controller.show_video()

    def show_history(self):
        """Opens the Emotion History window."""
        self.controller.show_history_page()
    def show_model_accuracy_page(self):
        """Opens the Model Accuracy window."""
        self.controller.show_model_accuracy_page()
