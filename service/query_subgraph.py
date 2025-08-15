import os
import sys
from typing import Any

import pandas as pd
from pymongo.synchronous.database import Database

sys.path.append(os.path.dirname(sys.path[0]))

from itertools import chain

from config import MongoDBConfig
from databases.mongodb import MongoDB
from utils.logger_utils import get_logger


def get_num_add(list_of_edges):
    all_values = list(
        chain.from_iterable(map(lambda d: (d["from"], d["to"]), list_of_edges))
    )
    unique_values = set(all_values)
    total_unique_values = len(unique_values)
    return total_unique_values


def get_vertices(edges):
    unique_values = set()
    for dictionary in edges:
        unique_values.update(dictionary.values())
    unique_values_list = list(unique_values)
    return unique_values_list


def preprocess_subgraph(subgraph):
    """
    Preprocess subgraph: get subgraph <= 100 vertices
    """
    subgraph.rename(columns={"address": "X_address"}, inplace=True)
    subgraph["NumAddress"] = subgraph["edges"].apply(get_num_add)
    filterr = subgraph[subgraph["NumAddress"] <= 200]
    filterr["vertices"] = filterr["edges"].apply(get_vertices)
    filterr.drop_duplicates("X_address", inplace=True)
    filterr.reset_index(inplace=True)
    filterr.drop("index", axis=1, inplace=True)
    return filterr


def query_subgraph(chain: str, radius: int):
    """
    Query and preprocess the subgraph
    """
    logger = get_logger(f"query subgraph of {chain}")
    logger.info(f"Querying subgraph for chain: {chain} with radius: {radius}")
    client: MongoDB = MongoDB(connection_url=MongoDBConfig.CONNECTION_URL)
    db: Database[Any] = client._db
    collection = db[f"subgraph_{chain}_{radius}"]
    # cursor = collection.find()
    cursor = collection.find()
    df_lst: list[pd.DataFrame] = []
    for re in cursor:
        df_sub: pd.DataFrame = pd.DataFrame([re])
        df_lst.append(df_sub)
    subgraph = pd.concat(df_lst)
    prep_subgraph = preprocess_subgraph(subgraph)
    return prep_subgraph
