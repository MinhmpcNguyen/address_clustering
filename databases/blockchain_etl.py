# import pymongo
# from pymongo import MongoClient

# from config import BlockchainETLConfig
# from constants.blockchain_etl_constants import BlockchainETLCollections
# from utils.logger_utils import get_logger

# logger = get_logger("Blockchain ETL")


# class BlockchainETL:
#     def __init__(self, connection_url=str, db_prefix=""):

#         self.connection = MongoClient(connection_url.split("@")[-1])

#         db_name: str = db_prefix + "_" + BlockchainETLConfig.DATABASE if db_prefix else BlockchainETLConfig.DATABASE


#         self.mongo_db = self.connection[db_name]

#         self.block_collection = self.mongo_db[BlockchainETLCollections.blocks]
#         self.transaction_collection = self.mongo_db[
#             BlockchainETLCollections.transactions
#         ]
#         self.collector_collection = self.mongo_db[BlockchainETLCollections.collectors]

#     def get_blocks_in_range(self, start_block, end_block, projection: list = None):
#         filter_ = {"number": {"$gte": start_block, "$lte": end_block}}
#         if not projection:
#             cursor = self.block_collection.find(filter_).batch_size(1000)
#         else:
#             cursor = self.block_collection.find(
#                 filter=filter_, projection=projection
#             ).batch_size(1000)
#         return cursor

#     def get_latest_tx_block_number(self) -> int:
#         _latest_tx_cursor = (
#             self.transaction_collection.find()
#             .sort("block_number", pymongo.DESCENDING)
#             .limit(1)
#         )
#         return list(_latest_tx_cursor)[0]["block_number"]

#     def get_contracts_deployers(
#         self, contract_addresses: list[str], from_block: int, to_block: int
#     ):
#         _filter = {
#             "block_number": {"$gte": from_block, "$lte": to_block},
#             "$or": [
#                 {"to_address": {"$in": contract_addresses}},
#                 {"receipt_contract_address": {"$in": contract_addresses}},
#             ],
#         }
#         _projection = ["from_address", "to_address", "receipt_contract_address"]
#         return self.transaction_collection.find(
#             filter=_filter, projection=_projection
#         ).batch_size(1000)
