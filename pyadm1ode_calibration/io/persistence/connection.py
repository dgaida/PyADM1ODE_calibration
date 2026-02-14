"""
Database Connection Management.

Handles SQLAlchemy engine creation and session factory configuration
for PostgreSQL database connections.
"""

import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """
    Configuration parameters for database connection.

    Attributes:
        host (str): Database host address.
        port (int): Port number.
        database (str): Database name.
        username (str): Username.
        password (str): Password.
        pool_size (int): SQLAlchemy pool size.
        max_overflow (int): SQLAlchemy max overflow.
        echo (bool): Whether to log SQL queries.
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "biogas"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class ConnectionManager:
    """
    Manages SQLAlchemy engine and session factory.

    Args:
        connection_string (Optional[str]): Full SQLAlchemy connection URL.
        config (Optional[DatabaseConfig]): Structured configuration object.
    """

    def __init__(self, connection_string: Optional[str] = None, config: Optional[DatabaseConfig] = None):
        if connection_string:
            self.connection_string = connection_string
        elif config:
            self.connection_string = (
                f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            )
        else:
            try:
                self.connection_string = self._get_from_env()
            except ValueError:
                raise ValueError("Either connection_string or config must be provided")

        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=config.pool_size if config else 5,
            max_overflow=config.max_overflow if config else 10,
            echo=config.echo if config else False,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _get_from_env(self) -> str:
        """
        Build connection string from standard environment variables.

        Returns:
            str: PostgreSQL connection string.
        """
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME")
        username = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD", "")

        if not all([database, username]):
            raise ValueError("Missing DB environment variables")

        return f"postgresql://{username}:{quote_plus(password)}@{host}:{port}/{database}"
