import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(sys.path[0]))
from collections import defaultdict
from itertools import chain

from multithread_processing.base_job import BaseJob

from databases.arangodb_klg import ArangoDB
from utils.logger_utils import get_logger
from utils.time_utils import TimeUtils


def get_list(list_of_list):
    """
    Change list of list to list
    """
    list_of_lists: list[str] = list(chain(*list_of_list))
    return list_of_lists


def preprocess(df, token_list):
    """
    Preprocess data after get time and amount feature: take the amount encoding and time histogram
    """
    df["time"] = df["time"].apply(get_list)
    df["valueInUSD"] = df["valueInUSD"].apply(get_list)
    df["valueInUSD"] = df["valueInUSD"].apply(np.mean)
    df["amount"] = df["amount"].apply(get_list)
    df["amount"] = df["amount"].apply(np.mean)
    filterCoins = df.copy()
    filterCoins["tokens"] = df["tokens"].apply(
        lambda x: x if x in token_list else "other_token"
    )
    filCoins = (
        filterCoins.groupby(["_id", "address"])[["time", "valueInUSD"]]
        .agg(list)
        .reset_index()
    )
    filCoins["time"] = filCoins["time"].apply(lambda x: list(chain(*x)))
    filCoins["time"] = filCoins["time"].apply(TimeUtils.get_time_histogram)
    valueTransferDf = filterCoins.pivot_table(
        index=["_id", "address"],
        columns="tokens",
        values="valueInUSD",
        aggfunc="sum",
        fill_value=0,
    )
    valueTransferDf.reset_index(inplace=True)
    TransferDf = valueTransferDf.merge(
        filCoins[["_id", "address", "time"]], on=["_id", "address"], how="left"
    )
    return TransferDf


class TimeAmountExporterJob(BaseJob):
    def __init__(
        self,
        chain: str,
        chain_id: str,
        list_index: list,
        token_list: list,
        edge_df: pd.DataFrame,
        transaction_database: ArangoDB,
        saving_path: str,
        max_workers=4,
        batch_size=1,
    ):
        self.all_from_lst = (
            list()
        )  # list that store all the wallet sending information of all subgraphs
        self.all_to_lst = (
            list()
        )  # list that store all the wallet receiving information of all subgraph
        self.subgraph = list()  # list of subgraph
        self.EdgeQuery = set()  # set of edges that have already taken

        self.token_list = token_list
        self.chain = chain
        self.chain_id = chain_id
        self.edge_df = edge_df
        self.transaction_database = transaction_database
        self.saving_path = saving_path

        super().__init__(
            work_iterable=list_index, max_workers=max_workers, batch_size=batch_size
        )

    def _execute_batch(self, works):
        get_logger("start to get time amount features")
        for work in works:
            subgraphDict = dict()
            subgraphDict["_id"] = self.edge_df.iloc[work]["_id"]
            edges_set = set()
            for item in self.edge_df.iloc[work]["edges"]:
                merged_value = (
                    f"{self.chain}_transfers/{self.chain_id}_"
                    + item["from"]
                    + "_"
                    + item["to"]
                )
                edges_set.add(merged_value)

            edges = list(edges_set - self.EdgeQuery)  # get the edges of subgraph
            query = f"""
                FOR doc IN {self.chain}_transfers
                FILTER doc._id IN @edges
                RETURN doc
            """

            results = self.transaction_database.aql.execute(
                query, bind_vars={"edges": edges}
            )
            from_lst = []
            to_lst = []
            edge_lst = []

            for a in results:
                edgeDict = dict()
                edgeDict["_from"] = a["_key"].split("_")[1]
                edgeDict["_to"] = a["_key"].split("_")[2]
                edge_lst.append(edgeDict)
                self.EdgeQuery.add(a["_id"])
                lst_txt = a["_id"].split("/")[1].split("_")
                transformed_data = defaultdict(list)

                for tokens, time_data in a["tokenTransferLogs"].items():
                    times = []
                    amounts = []
                    values_in_usd = []
                    for time, values in time_data.items():
                        if values.get("valueInUSD") == None:
                            continue
                        elif values["valueInUSD"] == None:
                            continue
                        times.append(time)
                        amounts.append(values["amount"])
                        values_in_usd.append(values["valueInUSD"])
                    if (times == []) | (amounts == []) | (values_in_usd == []):
                        continue
                    transformed_data["tokens"].append(tokens)
                    transformed_data["time"].append(times)
                    transformed_data["amount"].append(amounts)
                    transformed_data["valueInUSD"].append(values_in_usd)

                transformed_dict = dict(transformed_data)
                df = pd.DataFrame(transformed_dict)
                dfSingleFrom = df.copy()
                dfSingleTo = df.copy()
                dfSingleFrom["address"] = lst_txt[1]
                dfSingleFrom["edgeId"] = a["_id"]

                dfSingleTo["address"] = lst_txt[2]
                dfSingleTo["edgeId"] = a["_id"]
                from_lst.append(dfSingleFrom)
                to_lst.append(dfSingleTo)
            subgraphDict["vertices"] = edge_lst
            subgraph = pd.DataFrame(subgraphDict)
            try:
                self.all_from_lst.append(pd.concat(from_lst, axis=0))
                self.all_to_lst.append(pd.concat(to_lst, axis=0))
                self.subgraph.append(subgraph)

            except Exception:
                continue

    def _end(self):
        super()._end()
        logger = get_logger("Sucessful get time amount information")
        df_subgraph = pd.concat(self.subgraph, ignore_index=True)
        from_subgraph = pd.concat(self.all_from_lst, ignore_index=True)
        to_subgraph = pd.concat(self.all_to_lst, ignore_index=True)
        df_subgraph["edgeId"] = df_subgraph["vertices"].apply(
            lambda x: f"{self.chain}_transfers/{self.chain_id}"
            + "_"
            + x["_from"]
            + "_"
            + x["_to"]
        )

        from_subgraph = df_subgraph.merge(from_subgraph, on="edgeId", how="inner")
        to_subgraph = df_subgraph.merge(to_subgraph, on="edgeId", how="inner")
        from_df = (
            from_subgraph.groupby(["_id", "address", "tokens"])[
                ["edgeId", "time", "amount", "valueInUSD"]
            ]
            .agg(list)
            .reset_index()
        )  # get the wallet sending information in each subgraph
        to_df = (
            to_subgraph.groupby(["_id", "address", "tokens"])[
                ["edgeId", "time", "amount", "valueInUSD"]
            ]
            .agg(list)
            .reset_index()
        )  # get the wallet receiving information in each subgraph
        prep_from_path = f"{self.saving_path}/from_df.csv"
        prep_to_path = f"{self.saving_path}/to_df.csv"
        prep_from_df = preprocess(from_df, self.token_list)
        prep_to_df = preprocess(to_df, self.token_list)
        logger.info("successful preprocess time amount information")
        prep_from_df.to_csv(prep_from_path, index=False)
        prep_to_df.to_csv(prep_to_path, index=False)
