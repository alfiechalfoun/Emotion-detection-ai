import bcrypt

class PasswordManager:
    """A class to manage password hashing and verification."""

    @staticmethod
    def hash_password(password):
        """Hashes a password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password, hashed_password):
        """
        Verifies a password against a hashed password.
        
        Returns True if the password matches the hash, otherwise False.
        """
        return bcrypt.checkpw(password.encode(), hashed_password.encode())
