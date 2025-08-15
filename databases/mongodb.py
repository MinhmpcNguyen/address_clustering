from typing import Any, Generator, Iterator

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import InvalidOperation

from config import MongoDBConfig
from utils.logger_utils import get_logger

WALLETS_COL: str = "depositWallets"
logger = get_logger("MongoDB")


class MongoDB:
    def __init__(self, connection_url: str, chain_id: str = "") -> None:
        self.connection_url: str = connection_url.split("@")[-1]
        self.connection: MongoClient[Any] = MongoClient(host=connection_url)

        self._db: Database[Any] = self.connection[MongoDBConfig.DATABASE]
        self.lp_tokens_col: Collection[Any] = self._db["lpTokens"]
        self._deposit_wallets_col: Collection[Any] = self._db["depositWallets"]
        self._groups_col: Collection[Any] = self._db["groups"]
        self._deposit_users_col: Collection[Any] = self._db["depositUsers"]
        self._user_deposits_col: Collection[Any] = self._db["userDeposits"]
        self._transactions_col: Collection[Any] = self._db[f"{chain_id}_transactions"]
        self._token_transfers_col: Collection[Any] = self._db["token_transfers"]

    #######################
    #       Generals      #
    #######################

    def count_documents(self, col_name: str) -> int:
        return self._db[col_name].estimated_document_count()

    def get_documents(
        self, col_name: str, skip: int, limit: int
    ) -> Iterator[dict[str, Any]]:
        return self._db[col_name].find({}).skip(skip).limit(limit)

    def get_documents_by_ids(
        self, col_name: str, ids: list[str]
    ) -> Iterator[dict[str, str]]:
        return self._db[col_name].find({"_id": {"$in": ids}})

    #######################
    #   Deposit - Users   #
    #######################

    def get_deposit_number_of_users(
        self, skip: int = 0, limit: int = 1
    ) -> Generator[dict[str, Any], None, None]:
        pipeline: list[dict[str, int | dict[str, dict[str, str]]]] = [
            {"$project": {"numberOfUsers": {"$size": "$userWallets"}}}
        ]
        if skip:
            pipeline.append({"$skip": skip})
        if limit:
            pipeline.append({"$limit": limit})
        return self._deposit_users_col.aggregate(pipeline=pipeline)

    def get_deposit_wallet_by_users(
        self, chain_id: str, addresses: list[str]
    ) -> Iterator[dict[str, Any]]:
        _ids: list[str] = [f"{chain_id}_{address}" for address in addresses]
        _filter: dict[str, Any] = {"_id": {"$in": _ids}}
        _projection: dict[str, int] = {"_id": 1, "depositWallets": 1, "address": 1}
        return self._user_deposits_col.find(_filter, _projection)

    #######################
    #  Project deployers  #
    #######################

    def update_project_deployers(
        self, chain_id: str, project_deployers: dict[str, list[str]]
    ) -> None:
        bulk_updates: list[UpdateOne] = []
        for project, deployers in project_deployers.items():
            _id: str = f"{chain_id}_{project}"
            bulk_updates.append(
                UpdateOne(
                    filter={"_id": _id},
                    update={
                        "$set": {"project": project, "chainId": chain_id},
                        "$addToSet": {"deployers": {"$each": deployers}},
                    },
                    upsert=True,
                )
            )
        try:
            self._db["projectDeployers"].bulk_write(bulk_updates)
        except InvalidOperation as ex:
            _message: str = ex.args[0]
            if _message != "No operations to execute":
                raise ex
