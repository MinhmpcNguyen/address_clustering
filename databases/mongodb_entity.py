from pymongo import MongoClient
from pymongo.synchronous.cursor import Cursor

from config import MongoDBEntityConfig
from constants.coingecko_constants import CoingeckoConstant
from utils.logger_utils import get_logger

logger = get_logger("MongoDB Entity")


class MongoDBEntity:
    def __init__(self, connection_url: str):
        self.connection_url: str = connection_url.split("@")[-1]
        self.connection = MongoClient(connection_url)

        self._db = self.connection[MongoDBEntityConfig.DATABASE]
        self._config_col = self._db["configs"]
        self._multichain_wallets_col = self._db["multichain_wallets"]
        self._smart_contracts_col = self._db["smart_contracts"]

    # def get_native_token_price_change_logs(self, chain_id) -> Dict:
    #     _filter = {'_id': f"{chain_id}_{NATIVE_TOKENS[chain_id]}"}
    #     _projection = ['priceChangeLogs']
    #     return self._smart_contracts_col.find_one(filter=_filter, projection=_projection)

    def get_price_change_logs(self, chain_id, token_addresses):
        _token_ids = [f"{chain_id}_{address}" for address in token_addresses]
        _filter = {"_id": {"$in": _token_ids}, "priceChangeLogs": {"$exists": 1}}
        _projection = ["priceChangeLogs"]
        return self._smart_contracts_col.find(filter=_filter, projection=_projection)

    def get_top_token(self, chain_id: str) -> list[str]:
        """
        Return top 2000 tokens having the highest Market Cap from Coingecko
        """
        cursor: Cursor[dict[str, str]] = self._smart_contracts_col.find(
            {"idCoingecko": {"$exists": True}, "chainId": chain_id}, {"address": 1}
        )
        address_lst: list[str] = []
        for address in cursor[: CoingeckoConstant.TOP_MARKETCAP_COINGECKO]:
            address_lst.append(address["address"])

        return address_lst
