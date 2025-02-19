import sqlite3
from datetime import datetime


def create_connection(db_file):
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """Create the users table if it doesn't exist."""
    try:
        sql_create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,  -- UNIQUE constraint
            password TEXT NOT NULL
        );
        """

        # Emotions Table
        sql_create_emotions_table = """
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT NOT NULL,
            confidence FLOAT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """

        cursor = conn.cursor()
        cursor.execute(sql_create_users_table)
        cursor.execute(sql_create_emotions_table)
        conn.commit()
    except sqlite3.Error as e:
        print(e)

def insert_user(conn, first_name, last_name, email, username, password):
    """Insert a new user into the users table."""
    try:
        sql_insert_user = """
        INSERT INTO users (first_name, last_name, email, username, password)
        VALUES (?, ?, ?, ?, ?);
        """
        cursor = conn.cursor()
        cursor.execute(sql_insert_user, (first_name, last_name, email, username, password))
        conn.commit()
    except sqlite3.Error as e:
        print("Error inserting user:", e)


def log_emotion(conn, user_id, emotion, confidence):
    """Log an emotion detection event into the database."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql_insert_emotion = """
        INSERT INTO emotions (user_id, timestamp, emotion, confidence)
        VALUES (?, ?, ?, ?);
        """
        cursor = conn.cursor()
        cursor.execute(sql_insert_emotion, (user_id, timestamp, emotion, confidence))
        conn.commit()
    except sqlite3.Error as e:
        print("Error logging emotion:", e)
    
def get_emotion_history_by_date(conn, user_id, selected_date):
    """Retrieve the emotion history for a specific date."""
    sql_query = """
    SELECT timestamp, emotion, confidence FROM emotions
    WHERE user_id = ? AND DATE(timestamp) = ?
    ORDER BY timestamp DESC;
    """
    cursor = conn.cursor()
    cursor.execute(sql_query, (user_id, selected_date))
    return cursor.fetchall()  # Returns a list of tuples



def is_username_available(conn, username):
    """Check if a username is available."""
    sql_query = "SELECT 1 FROM users WHERE username = ? LIMIT 1;"
    cursor = conn.cursor()
    cursor.execute(sql_query, (username,))
    return cursor.fetchone() is None


def getname(conn, username):
    if not username:
        return "Guest"
    else:
            
        """Get the first name of the user by their username."""
        sql_query = "SELECT first_name FROM users WHERE username = ?;"
        cursor = conn.cursor()
        cursor.execute(sql_query, (username,))
        result = cursor.fetchone()

        if result is None:
            # Handle the case where the username does not exist
            raise ValueError(f"Username '{username}' not found in the database.")
        return result[0]
    
def get_user_id(conn, username):
    """Retrieve the user_id from the username."""
    sql_query = "SELECT id FROM users WHERE username = ?;"
    cursor = conn.cursor()
    cursor.execute(sql_query, (username,))
    result = cursor.fetchone()
    
    if result:
        return result[0]  # Return user_id
    else:
        return None

