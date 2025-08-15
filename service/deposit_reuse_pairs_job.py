import time
from itertools import permutations

import pandas as pd
from multithread_processing.base_job import BaseJob
from pandas import DataFrame

from constants.network_constants import Chains
from databases.arangodb_klg import ArangoDB
from utils.logger_utils import get_logger

ARANGO_BATCH_SIZE = 50000


class DepositReusePairJob:
    def __init__(
        self,
        chain: str,
        refresh_number_sent_received: bool = True,
        path: str = "data/deposit_reuse_pairs.csv",
        max_workers: int = 2,
        batch_size: int = 1000,
    ):
        self.logger = get_logger("Deposit Reuse Pairs Job")
        self.chain_name = chain
        self.chain_id = Chains.mapping.get(chain, None)
        if not self.chain_id:
            raise ValueError(f"Invalid chain name: {chain}")

        self.saving_path = path

        self.refresh_number_sent_received = refresh_number_sent_received

        self.max_workers = max_workers
        self.batch_size = batch_size

        self.arango = ArangoDB(prefix=self.chain_name)

        self.deposit_documents: list[dict] = []
        self.deposit_user_df: DataFrame = DataFrame()
        self.pairs_df: DataFrame = DataFrame()

    def run(self):
        self._get_deposit_wallets()
        self._get_users_df()
        self._generate_pairs()

    def _get_deposit_wallets(self):
        self.logger.info("Getting deposit accounts...")

        try:
            deposits_cursor = self.arango.get_deposit_wallets(
                batch_size=ARANGO_BATCH_SIZE
            )
            number_deposits = deposits_cursor.count()
            self.logger.info(f"Number of deposit accounts: {number_deposits}")
        except Exception:
            self.logger.error(
                "❌ Error while fetching deposit accounts from ArangoDB", exc_info=True
            )
            raise

        _count = 0
        while True:
            try:
                self.logger.info("Filtering of deposit accounts...")

                if not self.refresh_number_sent_received:
                    deposits_data: list[dict] = list(deposits_cursor.batch())
                else:
                    refresh_job = UpdateNumberSentReceivedJob(
                        addresses=[doc["address"] for doc in deposits_cursor.batch()],
                        chain_id=self.chain_id,
                        chain_name=self.chain_name,
                        arango=self.arango,
                        batch_size=1000,
                        max_workers=4,
                    )
                    deposits_data: list[dict] = refresh_job.run()

                _filtered_deposits_data = self._filter_deposit_wallet(
                    deposit_wallets=deposits_data
                )
                self.deposit_documents.extend(_filtered_deposits_data)

                deposits_cursor.batch().clear()
                _count += 1
                self.logger.info(
                    f"Filtered deposit accounts {_count * ARANGO_BATCH_SIZE} / {number_deposits}. "
                    f"PROGRESS: {_count * ARANGO_BATCH_SIZE / number_deposits * 100:.2f} %"
                )

                if deposits_cursor.has_more():
                    try:
                        deposits_cursor.fetch()
                    except Exception:
                        self.logger.error(
                            f"Error while fetching next batch from ArangoDB at batch #{_count + 1}",
                            exc_info=True,
                        )
                        raise
                else:
                    break

            except Exception:
                self.logger.error(
                    f"Error while processing deposit batch #{_count + 1}", exc_info=True
                )
                raise

        self.logger.info(f"Get {len(self.deposit_documents)} deposit accounts")

    @staticmethod
    def _filter_deposit_wallet(
        deposit_wallets: list[dict],
        hot_wallets_limit: int = 3,
        user_wallets_limit: int = 20,
    ) -> list[dict]:
        return [
            d
            for d in deposit_wallets
            if (
                d["numberSent"] < hot_wallets_limit
                and d["numberReceived"] < user_wallets_limit
            )
        ]

    def _get_users_df(self):
        deposit_addresses = [doc["address"] for doc in self.deposit_documents]
        self.logger.info(
            f"Getting users accounts from {len(deposit_addresses)} deposit accounts"
        )
        job = _GetUsersFromDepositsJob(
            arango=self.arango,
            chain_name=self.chain_name,
            chain_id=self.chain_id,
            deposit_addresses=deposit_addresses,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
        )
        _users_list: list[dict] = job.run()

        self.deposit_user_df = pd.DataFrame.from_records(
            [
                {
                    "deposit_address": datum["deposit_address"].split("_")[-1],
                    "user_address": datum["user_address"].split("_")[-1],
                }
                for datum in _users_list
            ]
        )

        self._filter_users_with_more_than_one_deposit()

        self._filter_bot_users()

        self.logger.info(
            f"Number of deposits: {self.deposit_user_df['deposit_address'].nunique()}"
        )
        self.logger.info(
            f"Number of users: {self.deposit_user_df['user_address'].nunique()}"
        )
        self.logger.info(f"Deposit - User:\n{self.deposit_user_df.describe()}")

    def _filter_users_with_more_than_one_deposit(self):
        self.logger.info("Filtering users with more than one deposit...")
        user_addr_counts = self.deposit_user_df["user_address"].value_counts()
        user_mask = self.deposit_user_df["user_address"].isin(
            user_addr_counts.index[user_addr_counts < 2]
        )
        self.deposit_user_df = self.deposit_user_df[user_mask]
        self.deposit_user_df = self.deposit_user_df.sort_values(
            "deposit_address"
        ).reset_index(drop=True)

    def _filter_bot_users(self, receivers_from_user_limit: int = 100) -> None:
        self.logger.info("Filtering bot users...")
        if self.refresh_number_sent_received:
            _job = UpdateNumberSentReceivedJob(
                addresses=self.deposit_user_df["user_address"].tolist(),
                chain_name=self.chain_name,
                chain_id=self.chain_id,
                arango=self.arango,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
            )
            _user_documents = _job.run()
        else:
            _user_documents = self.arango.get_vertices_by_addresses(
                self.deposit_user_df["user_address"].tolist(), batch_size=10000
            )

        _user_receiver_map = {
            doc["address"]: doc["numberSent"] for doc in _user_documents
        }

        self.deposit_user_df["n_receivers"] = self.deposit_user_df["user_address"].map(
            _user_receiver_map
        )
        self.deposit_user_df = self.deposit_user_df[
            self.deposit_user_df["n_receivers"] < receivers_from_user_limit
        ]
        self.deposit_user_df.drop(columns=["n_receivers"], inplace=True)
        self.deposit_user_df = self.deposit_user_df.sort_values(
            "deposit_address"
        ).reset_index(drop=True)

    def _generate_pairs(self):
        deposit_user_mapping = (
            self.deposit_user_df.groupby("deposit_address")["user_address"]
            .agg(list)
            .reset_index()
        )

        pairs_list = []
        for addresses in deposit_user_mapping["user_address"]:
            _pairs = list(permutations(addresses, 2))
            pairs_list.extend(_pairs)
        self.pairs_df = pd.DataFrame(pairs_list, columns=["X_address", "SubX_address"])
        self.logger.info(f"Pairs df:\n{self.pairs_df.describe()}")
        self.pairs_df.to_csv(self.saving_path)


class UpdateNumberSentReceivedJob(BaseJob):
    def __init__(
        self,
        addresses: list[str],
        chain_name: str,
        chain_id: str,
        arango: ArangoDB,
        batch_size: int = 10000,
        max_workers: int = 8,
    ):
        keys = [f"{chain_id}_{addr}" for addr in addresses]
        super().__init__(keys, batch_size, max_workers)
        self.arango = arango
        self.chain_name = chain_name
        # self.addresses_with_number_sent_received: dict[str, dict] = {}
        self.addresses_with_number_sent_received: list[dict] = []

    def _execute_batch(self, works):
        _time = int(time.time())
        _query = f"""
        FOR k in {works}
            LET id = CONCAT('{self.chain_name}_addresses/', k)
            LET n_to = (FOR v IN 1..1 OUTBOUND id
                GRAPH {self.chain_name}_transfers_graph
                COLLECT WITH COUNT INTO n
                RETURN n)
            LET n_from = (FOR v IN 1..1 INBOUND id
                GRAPH {self.chain_name}_transfers_graph
                COLLECT WITH COUNT INTO n
                RETURN n)
            UPDATE {{_key: k, numberSent: n_to[0], numberReceived: n_from[0], lastUpdatedAt:{_time} }}
                IN {self.chain_name}_addresses
            LET updated = NEW
            RETURN {{
                '_key': updated._key,
                'address': updated.address,
                'numberSent': updated.numberSent,
                'numberReceived': updated.numberReceived
            }}
        """
        new_data = self.arango.query(_query, batch_size=ARANGO_BATCH_SIZE)
        self.addresses_with_number_sent_received.extend(list(new_data))

    def run(self) -> list[dict]:
        super().run()
        return self.addresses_with_number_sent_received


class _GetUsersFromDepositsJob(BaseJob):
    def __init__(
        self,
        arango: ArangoDB,
        chain_id: str,
        chain_name: str,
        deposit_addresses,
        batch_size: int,
        max_workers: int,
    ):
        super().__init__(
            work_iterable=deposit_addresses,
            batch_size=batch_size,
            max_workers=max_workers,
        )

        self.chain_id = chain_id
        self.chain_name = chain_name
        self.arango = arango

        self._records: list[dict] = list()

    def _execute_batch(self, works):
        _deposit_address_ids = [
            f"{self.chain_name}_addresses/{self.chain_id}_{addr}" for addr in works
        ]
        _query = f"""
            FOR edge IN {self.chain_name}_transfers
            FILTER edge._to IN {_deposit_address_ids}
            RETURN {{
                'deposit_address': edge._to,
                'user_address': edge._from
            }}
        """
        returned_list = list(self.arango.query(query=_query))
        self._records.extend(returned_list)

    def run(self):
        super().run()
        return self._records
