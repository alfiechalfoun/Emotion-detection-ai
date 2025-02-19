import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFormLayout, QLineEdit, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
)
from sqlite3 import IntegrityError  # Import IntegrityError
from database import create_connection, create_table, insert_user, is_username_available
from hashedpasword import PasswordManager

class SignUpWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Sign Up")

        # --- Create the form fields ---
        self.first_name_edit = QLineEdit()
        self.last_name_edit = QLineEdit()
        self.email_edit = QLineEdit()
        self.username_edit = QLineEdit()

        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)

        self.confirm_password_edit = QLineEdit()
        self.confirm_password_edit.setEchoMode(QLineEdit.Password)

        # --- Create a button to show/hide password ---
        self.show_password_button = QPushButton("Show Password")
        self.show_password_button.setCheckable(True)
        self.show_password_button.clicked.connect(self.toggle_password_visibility)
        
        # Password layout with "Show Password" button
        password_layout = QHBoxLayout()
        password_layout.addWidget(self.password_edit)
        password_layout.addWidget(self.show_password_button)

        # --- Layout for form fields ---
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("First Name:"), self.first_name_edit)
        form_layout.addRow(QLabel("Last Name:"), self.last_name_edit)
        form_layout.addRow(QLabel("Email:"), self.email_edit)
        form_layout.addRow(QLabel("Username:"), self.username_edit)
        form_layout.addRow(QLabel("Password:"), password_layout)
        form_layout.addRow(QLabel("Confirm Password:"), self.confirm_password_edit)
        form_layout.setContentsMargins(10, 10, 10, 10)  # Adjusts the margins around the layout
        form_layout.setSpacing(5)  # Reduces the space between rows

        # --- Create Sign Up button and connect it ---
        self.sign_up_button = QPushButton("Sign Up")
        self.sign_up_button.clicked.connect(self.handle_sign_up)


        # Horizontal layout for navigation buttons
        self.nav_buttons_layout = QHBoxLayout()

        # Add "Back" button 
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.goto_back)
        self.nav_buttons_layout.addWidget(self.back_button)

        # Add "login" button 
        self.login_button = QPushButton("login")
        self.login_button.clicked.connect(self.goto_login)
        self.nav_buttons_layout.addWidget(self.login_button)

        # --- Add everything to main layout ---
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.sign_up_button)
        main_layout.addLayout(self.nav_buttons_layout)

        # Set layout
        self.setLayout(main_layout)

        # --- Set up database ---
        self.db_file = "users.db"  
        self.conn = create_connection(self.db_file)
        create_table(self.conn)

    def toggle_password_visibility(self):
        """Toggle the visibility of the password and confirm password fields."""
        if self.show_password_button.isChecked():
            self.password_edit.setEchoMode(QLineEdit.Normal)
            self.show_password_button.setText("Hide Password")
        else:
            self.password_edit.setEchoMode(QLineEdit.Password)
            self.show_password_button.setText("Show Password")

    def handle_sign_up(self):
        # Get the user input from the form
        first_name = self.first_name_edit.text().strip()
        last_name = self.last_name_edit.text().strip()
        email = self.email_edit.text().strip()
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        confirm_password = self.confirm_password_edit.text().strip()

        # Basic validation
        if not all([first_name, last_name, email, username, password, confirm_password]):
            QMessageBox.warning(self, "Error", "Please fill in all fields.")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        
        if password == username:
            QMessageBox.warning(self, "Error", "Password cannot be the same as username.")
            return
        
        if not is_username_available(self.conn, username):
            QMessageBox.warning(self, "Error", "Username already exists. Please choose a different username.")
            return

        import re
        password_regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"

        if not re.match(email_regex, email):
            QMessageBox.warning(self, "Error", "Invalid email format.")
            return

        if not re.match(password_regex, password):
            QMessageBox.warning(self, "Error", "Password must contain at least 8 characters, including one uppercase letter, one lowercase letter, one number, and one special character.")
            return
        
        try:
            hashed_password = PasswordManager.hash_password(password)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Password hashing failed: {e}")
            return

        # Insert into database
        try:
            insert_user(self.conn, first_name, last_name, email, username, hashed_password)
            QMessageBox.information(self, "Success", "User account created successfully!")
            # Clear fields after successful sign-up
            self.first_name_edit.clear()
            self.last_name_edit.clear()
            self.email_edit.clear()
            self.username_edit.clear()
            self.password_edit.clear()
            self.confirm_password_edit.clear()
            self.controller.show_login_page()  # Navigate to login page after successful sign-up
        except IntegrityError:
            QMessageBox.warning(self, "Error", "Username already exists. Please choose a different username.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create user: {e}")

    def goto_back(self):
        if self.controller:
            self.controller.show_welcome_screen()
        
    def goto_login(self):
        if self.controller:
            self.controller.show_login_page()

    def closeEvent(self, event):
        if self.conn:
            self.conn.close()
        super().closeEvent(event)

