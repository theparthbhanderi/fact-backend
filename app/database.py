import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use SQLite for local storage
DB_DIR = os.path.dirname(os.path.abspath(__file__))
# Put the DB in the backend root directory (parent of app/)
DB_PATH = os.path.join(os.path.dirname(DB_DIR), "factchecker.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create the SQLAlchemy engine
# connect_args={"check_same_thread": False} is needed only for SQLite + FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
