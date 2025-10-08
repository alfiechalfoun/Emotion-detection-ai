from PyQt5.QtWidgets import QApplication, QStackedWidget
from openning_screen import StartupPage 
from loginpage import LoginPage
from sinup import SignUpWindow
from mainpage import MainPage
from RunVideo import RunVideo
from database import create_connection, get_user_id
from history import HistoryWindow
from ModleAcuracy import ModelAccuracyPage
import sqlite3
import sys

class AppController(QStackedWidget):
    def __init__(self):
        self.conn = create_connection('users.db')
        self.current_user = None
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create instances of screens
        self.welcome_screen = StartupPage(self)
        self.login_page = LoginPage(self)
        self.signup_page = SignUpWindow(self)
        self.main_page = MainPage(self, self.current_user)

        # Add widgets to the stack  
        self.addWidget(self.welcome_screen)
        self.addWidget(self.login_page)
        self.addWidget(self.signup_page)
        self.addWidget(self.main_page)

        # Set the initial screen
        self.setCurrentWidget(self.welcome_screen)  
        self.setWindowTitle("Emotion Detector")
        self.resize(450, 318)

    def show_video(self):
        self.video = RunVideo(self, self.current_user)
        self.addWidget(self.video)
        self.setCurrentWidget(self.video)

    def show_login_page(self):
        self.setCurrentWidget(self.login_page)
        self.resize(450, 318)


    def show_signup_page(self):
        self.setCurrentWidget(self.signup_page)
        self.resize(450, 318)

    def show_welcome_screen(self):
        self.current_user = None
        self.setCurrentWidget(self.welcome_screen)
        self.resize(450, 318)

    def show_model_accuracy_page(self):
        self.modle_accuracy = ModelAccuracyPage(self)
        self.addWidget(self.modle_accuracy)
        self.setCurrentWidget(self.modle_accuracy)
    
    def show_history_page(self):
        """Opens the History Window with the correct database connection."""
        user_id = get_user_id(self.conn, self.current_user)  # Get the user ID
        self.history = HistoryWindow(self, self.conn, user_id)  
        self.addWidget(self.history)
        self.setCurrentWidget(self.history)
        self.resize(450, 318)


    # changed to make a new instance inored to updat the name
    def show_main_page(self):
        self.main_page = MainPage(self, self.current_user)  
        self.addWidget(self.main_page)
        self.setCurrentWidget(self.main_page)
        self.resize(450, 318)
    
    def set_current_user(self, username):
        self.current_user = username

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = AppController()
    controller.show()
    sys.exit(app.exec_())