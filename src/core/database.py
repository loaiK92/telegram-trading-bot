# src/core/database.py
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Define the database file path
DATABASE_URL = "sqlite:///./bot_database.db"

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} # Required for SQLite with Telegram bot
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for our models to inherit from
Base = declarative_base()

# Define the User model (table)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, nullable=False, index=True)
    language = Column(String, default="en")

# Function to create the database and table
def init_db():
    Base.metadata.create_all(bind=engine)

def get_user_language(telegram_id: int) -> str:
    """Gets the user's language, defaulting to 'en' if not found."""
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    db.close()
    if user:
        return user.language
    return 'en'

def set_user_language(telegram_id: int, language_code: str):
    """Sets or updates the user's language."""
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if user:
        user.language = language_code
    else:
        user = User(telegram_id=telegram_id, language=language_code)
        db.add(user)
    db.commit()
    db.close()