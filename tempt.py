import random

import pandas as pd
from IPython.utils import io
from multithread_processing.base_job import BaseJob
from utils.embedding_utils import EmbeddingUtils
from utils.logger_utils import get_logger


def get_label(x):
    """
    Return True if label == True, return False if label are not True
    """
    if x == True:
        return True
    else:
        return False


def apply_embedding_similarity(sub_df):
    sub_df["X_Diff2VecEmbedding"] = sub_df["X_Diff2VecEmbedding"].apply(
        lambda x: EmbeddingUtils.get_embedding_list(x)
    )
    sub_df["SubX_Diff2VecEmbedding"] = sub_df["SubX_Diff2VecEmbedding"].apply(
        lambda x: EmbeddingUtils.get_embedding_list(x)
    )
    sub_df["Diff2_Vec_Simi"] = sub_df.apply(
        lambda x: EmbeddingUtils.diff_cosine(x), axis=1
    )
    return sub_df


class ProcessTrainingDataset(BaseJob):
    """
    #Preprocess training dataset
    """

    def __init__(
        self,
        training_dataset: pd.DataFrame,
        list_index: list,
        logger: get_logger,
        saving_path: str,
        compute_embedding_similarity: bool = True,
        max_workers=2,
        batch_size=1,
    ):
        self.training_dataset = training_dataset
        self.prep_df_lst = list()  # List of preprocessed dataset
        self.logger = logger
        self.saving_path = saving_path
        self.compute_embedding_similarity = compute_embedding_similarity
        super().__init__(
            work_iterable=list_index, max_workers=max_workers, batch_size=batch_size
        )

    def _execute_batch(self, works):
        try:
            with io.capture_output() as captured:
                sub = self.training_dataset.copy()
                sub_df = sub.iloc[works]

                if self.compute_embedding_similarity:
                    sub_df = apply_embedding_similarity(sub_df)

                sub_df[[f"X_Time{i}" for i in range(24)]] = sub_df.X_Time.apply(
                    pd.Series
                )
                sub_df[[f"SubX_Time{i}" for i in range(24)]] = sub_df.SubX_Time.apply(
                    pd.Series
                )
                self.prep_df_lst.append(sub_df)
        except Exception as e:
            self.logger.exception(f"Can't process df: {e}")

    def _end(self):
        super()._end()
        self.logger.info("Generate time columns and embedding similarity successful")
        prep_df = pd.concat(self.prep_df_lst)
        df_analyzing = prep_df.pivot_table(
            index="_id", columns="Label", aggfunc="size", fill_value=0
        ).reset_index()
        lst_id = df_analyzing[df_analyzing[True] >= 1]._id.unique().tolist()
        dfPrep = prep_df[prep_df["_id"].isin(lst_id)]
        dfPrep = dfPrep.sort_index(axis=1)
        usr_lst = prep_df["X_address"].unique().tolist()
        self.logger.info("Spliting and saving dataset ...")
        random_elements = random.sample(usr_lst, int(len(usr_lst) * 0.9))
        train_data = dfPrep[dfPrep["X_address"].isin(random_elements)].reset_index(
            drop=True
        )
        test_data = dfPrep[~dfPrep["X_address"].isin(random_elements)].reset_index(
            drop=True
        )

        drop_cols = [
            "_id",
            "SubX_Time",
            "SubX_From_time",
            "SubX_To_time",
            "X_From_time",
            "X_To_time",
            "X_Time",
            "X_address",
            "SubX_address",
        ]

        if self.compute_embedding_similarity:
            drop_cols += ["X_Diff2VecEmbedding", "SubX_Diff2VecEmbedding"]

        train_data.drop(drop_cols, axis=1, inplace=True)
        test_data.drop(drop_cols, axis=1, inplace=True)

        train_data.to_csv(f"{self.saving_path}/train_data.csv", index=False)
        test_data.to_csv(f"{self.saving_path}/test_data.csv", index=False)
        self.logger.info("Successful")
