import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtCore import Qt

# Import the Main class from the main module
try:
    from run import Run as Main
except ImportError as e:
    print(f"Error importing Main: {e}")
    sys.exit(1)

class StartupPage(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Starting page")
        self.setGeometry(100, 100, 800, 600)  # Set window size

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout for the main page
        layout = QVBoxLayout()

        # Add a title label
        title_label = QLabel("Emotion Detector")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Add a button to start the main class
        start_button = QPushButton("Start as guest")
        start_button.setStyleSheet("font-size: 18px;")
        start_button.clicked.connect(self.start_main)
        layout.addWidget(start_button)

        # Add login and sign-up buttons in a horizontal layout
        button_layout = QHBoxLayout()

        # Login button
        login_button = QPushButton("Login")
        login_button.setStyleSheet("font-size: 18px;")
        login_button.clicked.connect(self.controller.show_login_page)
        button_layout.addWidget(login_button)

        # Sign Up button
        signup_button = QPushButton("Sign Up")
        signup_button.setStyleSheet("font-size: 18px;")
        signup_button.clicked.connect(self.controller.show_signup_page)
        button_layout.addWidget(signup_button)

        # Align buttons to the center of the horizontal layout
        button_layout.setAlignment(Qt.AlignCenter)

        # Add the button layout to the main vertical layout
        layout.addLayout(button_layout)

        # Set layout to central widget
        self.central_widget.setLayout(layout)

    def start_main(self):
        self.controller.show_video()

