import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.http import DefaultHTTPClient
from arango.result import Result
from config import ArangoDBConfig

from utils.logger_utils import get_logger
from utils.parser import get_connection_elements
from utils.retry_handler import retry_handler

logger = get_logger("ArangoDB")


class ArangoDB:
    def __init__(
        self, connection_url=None, database=ArangoDBConfig.DATABASE, prefix: str = None
    ):
        if not connection_url:
            connection_url = ArangoDBConfig.CONNECTION_URL
        _username, _password, _connection_url = get_connection_elements(connection_url)

        http_client = DefaultHTTPClient()
        http_client.REQUEST_TIMEOUT = 1000

        try:
            self.client = ArangoClient(hosts=_connection_url, http_client=http_client)
        except Exception as e:
            logger.exception(f"Failed to connect to ArangoDB: {_connection_url}: {e}")
            sys.exit(1)

        self.prefix = prefix
        self._db = self._get_db(database, _username, _password)

        self._addresses_col_name = f"{prefix}_addresses"
        self._transfers_col_name = f"{prefix}_transfers"
        self._transfers_graph_name = f"{prefix}_transfers_graph"

        self._addresses_col = self._get_collections(self._addresses_col_name)
        self._transfers_col = self._get_collections(self._transfers_col_name, edge=True)
        _transfers_graph_edge_definitions = [
            {
                "edge_collection": self._transfers_col_name,
                "from_vertex_collections": [self._addresses_col_name],
                "to_vertex_collections": [self._addresses_col_name],
            }
        ]
        self._transfers_graph = self._get_graph(
            graph_name=self._transfers_graph_name,
            edge_definitions=_transfers_graph_edge_definitions,
        )

    def _get_db(self, db_name, username, password):
        return self.client.db(db_name, username=username, password=password)

    def _get_graph(
        self, graph_name, edge_definitions, database: StandardDatabase = None
    ):
        if not database:
            database = self._db
        if not database.has_graph(graph_name):
            database.create_graph(graph_name, edge_definitions=edge_definitions)
        return database.graph(graph_name)

    def _get_collections(
        self, collection_name, database: StandardDatabase = None, edge=False
    ):
        if not database:
            database = self._db
        if not database.has_collection(collection_name):
            database.create_collection(collection_name, shard_count=20, edge=edge)
        return database.collection(collection_name)

    #####################
    #     Retrieve      #
    #####################
    def get_number_of_relationships(self):
        return len(self._transfers_col)

    def get_token_transfers(self, skip, limit):
        return self._transfers_col.all(skip=skip, limit=limit)

    def check_has_relationship(self, edge_key: str):
        edge_id = f"{self.prefix}_transfers/{edge_key}"
        try:
            assert edge_id in self._transfers_col
            return True
        except AssertionError:
            return False

    @retry_handler(retries_number=3, sleep_time=5)
    def get_vertices_by_addresses(
        self, addresses: list[str], batch_size: int = 10000
    ) -> Result:
        _query = f"""
            FOR v in {self.prefix}_addresses
            FILTER v.address IN {addresses}
            RETURN {{
                '_key': v._key, 
                'address': v.address, 
                'numberSent': v.numberSent, 
                'numberReceived': v.numberReceived
            }}
        """
        return self._db.aql.execute(query=_query, batch_size=batch_size, count=True)

    @retry_handler(retries_number=3, sleep_time=5)
    def get_deposit_wallets(self, batch_size: int = 10000) -> Result:
        _query = f"""
            FOR v in {self.prefix}_addresses
            FILTER v.wallet.depositWallet
            RETURN {{
                '_key': v._key, 
                'address': v.address, 
                'numberSent': v.numberSent, 
                'numberReceived': v.numberReceived
            }}
        """
        return self._db.aql.execute(query=_query, batch_size=batch_size, count=True)

    def get_tagged_wallets(self, tag: str, batch_size: int = 10000) -> Result:
        """
        Args:
            tag: 'hotWallet' or 'depositWallet'
            batch_size: for arango cursor
        """
        _query = f"""
            FOR v in {self.prefix}_addresses
            FILTER v.wallet.{tag}
            RETURN v
        """
        return self._db.aql.execute(query=_query, batch_size=batch_size, count=True)

    @staticmethod
    def _parse_id(id_string):
        return id_string.split("/")[-1]

    def query(self, query: str, batch_size=1000) -> Result:
        return self._db.aql.execute(query=query, batch_size=batch_size)
