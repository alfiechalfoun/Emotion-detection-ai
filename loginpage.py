from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QHBoxLayout
)
import sqlite3
from hashedpasword import PasswordManager
import sys

class LoginPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Login Page")
        self.setGeometry(100, 100, 300, 200)

        # Create the layout
        self.layout = QVBoxLayout()

        # Add a title label
        self.title_label = QLabel("Login to Your Account")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        # Username input
        self.username_label = QLabel("Username:")
        self.layout.addWidget(self.username_label)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.layout.addWidget(self.username_input)

        # Password input
        self.password_label = QLabel("Password:")
        self.layout.addWidget(self.password_label)
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)  # Mask the password
        self.layout.addWidget(self.password_input)

        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.handle_login)
        self.layout.addWidget(self.login_button)


        # Horizontal layout for navigation buttons
        self.nav_buttons_layout = QHBoxLayout()

        # Add "Back" button 
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.goto_back)
        self.nav_buttons_layout.addWidget(self.back_button)

                # Add "Sign Up" button 
        self.signup_button = QPushButton("Sign Up")
        self.signup_button.clicked.connect(self.goto_signup)
        self.nav_buttons_layout.addWidget(self.signup_button)

        # Add horizontal layout to main layout
        self.layout.addLayout(self.nav_buttons_layout)

        # Set layout
        self.setLayout(self.layout)


    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # Check if the username and password are correct
        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password.")
            return
        
        try:
            # Query the database to check if the username exists and fetch the hashed password
            sql_query = "SELECT password FROM users WHERE username = ?"
            cursor = self.controller.conn.cursor()
            cursor.execute(sql_query, (username,))
            result = cursor.fetchone()

            if result is None:
                QMessageBox.warning(self, "Login Failed", "Username does not exist.")
                return

            stored_hashed_password = result[0]

            # Compare the provided password with the stored hashed password
            if PasswordManager.verify_password(password, stored_hashed_password):
                QMessageBox.information(self, "Login Success", "Welcome!")
                self.controller.set_current_user(username)
                self.controller.show_main_page()
                self.username_input.clear()
                self.password_input.clear()
            else:
                QMessageBox.warning(self, "Login Failed", "Incorrect password.")
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"An error occurred: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")


    def goto_back(self):
        if self.controller:
            self.controller.show_welcome_screen()
        
    def goto_signup(self):
        if self.controller:
            self.controller.show_signup_page()
            
