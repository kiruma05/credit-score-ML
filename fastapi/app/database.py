# import os
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# DATABASE_URL = os.getenv("DATABASE_URL")
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()


import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("[CRITICAL] DATABASE_URL not set in environment.", file=sys.stderr)
    # Fail early to avoid running without a database connection
    raise RuntimeError("DATABASE_URL environment variable is missing!")

# PostgreSQL connection tuning (for Docker / MLflow setup)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,             # Detect dead connections
    pool_size=10,                   # Maintain small pool
    max_overflow=20,                # Allow burst of connections
    connect_args={"connect_timeout": 10},  # Prevent hanging if DB slow
)

# SQLAlchemy session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for ORM models
Base = declarative_base()

print(f"[INFO] Database engine initialized for {DATABASE_URL}")
