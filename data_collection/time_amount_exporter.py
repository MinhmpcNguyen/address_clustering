import pandas as pd

from config import MongoDBEntityConfig
from constants.network_constants import Chains
from constants.time_constants import TimeConstants
from databases.arangodb_klg import ArangoDB
from databases.mongodb_entity import MongoDBEntity
from service.query_subgraph import query_subgraph
from service.time_amount_exporter_service import TimeAmountExporterJob
from utils.logger_utils import get_logger

logger = get_logger("Time Amount Exporter Scheduler")


def time_amount_exporter(
    saving_path: str,
    chain: str = "ethereum",
    radius: int = 2,
    batch_size: int = TimeConstants.AN_HOUR,
    max_workers: int = 8,
):
    """Run the Time Amount Exporter job with given parameters."""
    chain = chain.lower()
    if chain not in Chains.mapping:
        raise ValueError(f"Chain '{chain}' is not supported")
    chain_id: str = Chains.mapping[chain]

    logger.info("Getting token list")
    token_list: list[str] = MongoDBEntity(
        connection_url=MongoDBEntityConfig.CONNECTION_URL
    ).get_top_token(chain_id=chain_id)

    logger.info("Starting query subgraph")
    arangodb: ArangoDB = ArangoDB(prefix=chain)

    prep_subgraph: pd.DataFrame = query_subgraph(chain, radius)
    idx: list[int] = list(prep_subgraph.index)

    logger.info("Starting export features")

    job: TimeAmountExporterJob = TimeAmountExporterJob(
        chain=chain,
        chain_id=chain_id,
        list_index=idx,
        token_list=token_list,
        edge_df=prep_subgraph,
        transaction_database=arangodb._db,
        saving_path=saving_path,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    job.run()
