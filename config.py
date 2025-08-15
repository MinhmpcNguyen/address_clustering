import os

from dotenv import load_dotenv

load_dotenv()


class ArangoDBConfig:
    HOST: str = os.environ.get("ARANGODB_HOST", "0.0.0.0")
    PORT: str = os.environ.get("ARANGODB_PORT", "8529")
    USERNAME: str = os.environ.get("ARANGODB_USERNAME", "root")
    PASSWORD: str = os.environ.get("ARANGODB_PASSWORD", "dev123")

    CONNECTION_URL: str = (
        os.getenv("ARANGODB_CONNECTION_URL")
        or f"arangodb@{USERNAME}:{PASSWORD}@http://{HOST}:{PORT}"
    )

    DATABASE: str = os.getenv("ARANGODB_DATABASE", "wallet_graph")
    GRAPH: str = "knowledge_graph"


class PostgresDBConfig:
    SCHEMA: str = os.environ.get("POSTGRES_SCHEMA", "public")
    TRANSFER_EVENT_TABLE: str = os.environ.get(
        "POSTGRES_TRANSFER_EVENT_TABLE", "transfer_event"
    )
    CONNECTION_URL: str = os.environ.get(
        "POSTGRES_CONNECTION_URL", "postgresql://user:password@localhost:5432/database"
    )


class BlockchainETLConfig:
    HOST: str = os.getenv("BLOCKCHAIN_ETL_HOST", "")
    PORT: str = os.getenv("BLOCKCHAIN_ETL_PORT", "")
    USERNAME: str = os.getenv("BLOCKCHAIN_ETL_USERNAME", "")
    PASSWORD: str = os.getenv("BLOCKCHAIN_ETL_PASSWORD", "")

    CONNECTION_URL: str = (
        os.getenv("BLOCKCHAIN_ETL_CONNECTION_URL")
        or f"mongodb://{USERNAME}:{PASSWORD}@{HOST}:{PORT}"
    )
    DATABASE: str = "blockchain_etl"
    DB_PREFIX: str = os.getenv("DB_PREFIX", "")


class MongoDBConfig:
    CONNECTION_URL: str = os.getenv(
        "MONGODB_CONNECTION_URL", "mongodb://131.153.202.197:28017"
    )
    DATABASE: str = os.getenv("MONGODB_DATABASE", "knowledge_graph")


class MongoDBEntityConfig:
    CONNECTION_URL: str = os.getenv(
        "MONGODB_ENTITY_CONNECTION_URL",
        "mongodb://klgReader:klgReaderEntity_910@178.128.85.210:27017,104.248.148.66:27017,103.253.146.224:27017/",
    )
    DATABASE: str = os.getenv("MONGODB_ENTITY_DATABASE", "knowledge_graph")


class MongoDBSmartContractConfig:
    CONNECTION_URL: str = os.getenv("MONGODB_SMARTCONTRACT_CONNECTION_URL", "")
    DATABASE: str = os.getenv("MONGODB_SMARTCONTRACT_DATABASE", "SmartContractLabel")
