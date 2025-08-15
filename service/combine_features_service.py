import random
from operator import add
from typing import List

import pandas as pd
from IPython.utils import io
from multithread_processing.base_job import BaseJob

from utils.embedding_utils import EmbeddingUtils
from utils.logger_utils import get_logger


def get_time_histogram(row):
    """
    Get time histogram of row 0 when merge to_df and from_df
    """
    if row == 0:
        row = [0] * 24
    return str(row)


def get_time(row):
    """
    Concat sending time and receiving time of a wallet
    """
    return list(map(add, row["From_time"], row["To_time"]))


def get_label(x: bool) -> bool:
    """
    Return True if label == True, return False if label are not True
    """
    if x == True:
        return True
    else:
        return False


def apply_embedding_similarity(sub_df: pd.DataFrame) -> pd.DataFrame:
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


def merge_from_to(df_from: pd.DataFrame, df_to: pd.DataFrame) -> pd.DataFrame:
    get_logger("Merging sending and receiving information ...")
    for cols in df_from.columns:
        if cols not in ["_id", "address"]:
            df_from.rename(columns={cols: f"From_{cols}"}, inplace=True)

    for cols in df_to.columns:
        if cols not in ["_id", "address"]:
            df_to.rename(columns={cols: f"To_{cols}"}, inplace=True)

    df: pd.DataFrame = df_from.merge(df_to, how="outer", on=["_id", "address"])
    assert df is not None, "Merge operation returned None"

    df = df.fillna(0)
    assert df is not None, "Fillna operation returned None"

    df = df.drop_duplicates()
    assert df is not None, "Drop duplicates operation returned None"

    df["To_time"] = df["To_time"].apply(get_time_histogram).apply(eval)
    df["From_time"] = df["From_time"].apply(get_time_histogram).apply(eval)
    df["Time"] = df.apply(lambda x: get_time(x), axis=1)
    get_logger("Sending and receiving information merged.")
    return df


def merge_embedding(df: pd.DataFrame, df_embedding: pd.DataFrame) -> pd.DataFrame:
    get_logger("Merging embedding information ...")
    df_embedding.rename(columns={"vertices": "address"}, inplace=True)
    df = df.merge(df_embedding, on=["_id", "address"], how="left").dropna()

    return df


def merge_pair_labels(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    # logger: get_logger,
    contract: pd.DataFrame = None,
) -> pd.DataFrame:
    get_logger("Merging pair labels ...")
    if contract is not None:
        contract.drop("Unnamed: 0", axis=1, inplace=True)
        df = df.merge(contract, on="address", how="inner")
        df = df[df["IsContract"] == False]
        df.drop("IsContract", axis=1, inplace=True)

    X_df: pd.DataFrame = df[df["_id"].apply(lambda x: x.split("_")[1]) == df["address"]]
    for col in X_df.columns:
        if col != "_id":
            X_df.rename(columns={col: f"X_{col}"}, inplace=True)
    X_lst: List[str] = X_df["X_address"].unique().tolist()

    SubX_df: pd.DataFrame = df[
        df["_id"].apply(lambda x: x.split("_")[1]) != df["address"]
    ]
    for col in SubX_df.columns:
        if col != "_id":
            SubX_df.rename(columns={col: f"SubX_{col}"}, inplace=True)

    pair_features: pd.DataFrame = SubX_df.merge(X_df, on="_id", how="outer").dropna()
    pair_filters: pd.DataFrame = pairs[pairs["X_address"].isin(X_lst)].copy()
    pair_filters["Label"] = True

    final_pair: pd.DataFrame = pair_features.merge(
        pair_filters, on=["X_address", "SubX_address"], how="left"
    )
    final_pair["Label"] = final_pair["Label"].apply(get_label)
    get_logger("Pair labels merged.")
    return final_pair


class ProcessTrainingDataset(BaseJob):
    """
    #Preprocess training dataset
    """

    def __init__(
        self,
        training_dataset: pd.DataFrame,
        list_index: List[int],
        logger: get_logger,
        saving_path: str,
        compute_embedding_similarity: bool = True,
        max_workers: int = 2,
        batch_size: int = 1,
    ):
        self.training_dataset: pd.DataFrame = training_dataset
        self.prep_df_lst: List[pd.DataFrame] = []  # List of preprocessed dataset
        self.logger = logger
        self.saving_path: str = saving_path
        self.compute_embedding_similarity: bool = compute_embedding_similarity
        super().__init__(
            work_iterable=list_index, max_workers=max_workers, batch_size=batch_size
        )

    def _execute_batch(self, works: List[int]):
        try:
            # Log kiểm tra training_dataset trước khi dùng
            # if not hasattr(self, "training_dataset"):
            #     self.logger.error("training_dataset attribute does not exist in this instance.")
            #     return
            if self.training_dataset is None:
                self.logger.error("training_dataset is None.")
                return
            if not isinstance(self.training_dataset, pd.DataFrame):
                self.logger.error(
                    f"training_dataset is not a DataFrame. Type: {type(self.training_dataset)}"
                )
                return

            self.logger.info(f"training_dataset shape: {self.training_dataset.shape}")

            with io.capture_output() as captured:
                sub: pd.DataFrame = self.training_dataset.copy()
                sub_df: pd.DataFrame = sub.iloc[works]

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

        # Gộp các DataFrame đã xử lý trước đó
        if not getattr(self, "prep_df_lst", None):
            self.logger.error("prep_df_lst is empty -> nothing to concatenate.")
            return

        prep_df: pd.DataFrame = pd.concat(self.prep_df_lst, ignore_index=True)
        self.logger.info(f"prep_df shape: {prep_df.shape}")
        if "_id" not in prep_df.columns or "Label" not in prep_df.columns:
            self.logger.error(
                f"Missing required columns in prep_df. Columns: {list(prep_df.columns)}"
            )
            return

        # Log phân bố Label ban đầu
        self.logger.info(f"Label dtype: {prep_df['Label'].dtype}")
        self.logger.info(
            f"Label value_counts: {prep_df['Label'].value_counts(dropna=False).to_dict()}"
        )

        # Chuẩn hóa Label sang boolean để pivot (không thay đổi cột Label gốc)
        true_set = {True, "True", "true", 1, "1"}
        false_set = {False, "False", "false", 0, "0"}

        def to_bool(v):
            if v in true_set:
                return True
            if v in false_set:
                return False
            return False  # mặc định False nếu giá trị lạ

        prep_df["Label_bool"] = prep_df["Label"].apply(to_bool)

        # Pivot để đếm số lượng theo nhãn boolean
        df_analyzing: pd.DataFrame = prep_df.pivot_table(
            index="_id", columns="Label_bool", aggfunc="size", fill_value=0
        ).reset_index()

        self.logger.info(
            f"df_analyzing columns (after pivot): {list(df_analyzing.columns)}"
        )
        if True not in df_analyzing.columns:
            df_analyzing[True] = 0

        # Lọc các _id có ít nhất 1 label True
        lst_id: List[str] = df_analyzing[df_analyzing[True] >= 1]._id.unique().tolist()
        self.logger.info(f"Number of ids with True>=1: {len(lst_id)}")

        # Lọc dữ liệu tương ứng; fallback nếu rỗng
        if len(lst_id) == 0:
            self.logger.warning(
                "No IDs with True label found -> fallback to using entire prep_df."
            )
            dfPrep: pd.DataFrame = prep_df.copy()
        else:
            dfPrep: pd.DataFrame = prep_df[prep_df["_id"].isin(lst_id)].copy()

        self.logger.info(f"dfPrep shape (after filter): {dfPrep.shape}")

        # Danh sách địa chỉ (từ dfPrep để phù hợp với filter)
        if "X_address" not in dfPrep.columns:
            self.logger.error("Missing 'X_address' in dfPrep. Cannot split by address.")
            return

        usr_lst: List[str] = dfPrep["X_address"].dropna().unique().tolist()
        n_addrs = len(usr_lst)
        self.logger.info(f"Unique X_address count: {n_addrs}")

        self.logger.info("Splitting and saving dataset ...")

        # Chia train/test theo địa chỉ (an toàn cho tập nhỏ)
        if n_addrs >= 2:
            k = max(1, min(n_addrs - 1, int(n_addrs * 0.9)))
            self.logger.info(f"Sampling {k} addresses out of {n_addrs} (~90%)")
            random_elements: List[str] = random.sample(usr_lst, k)
            train_data: pd.DataFrame = dfPrep[
                dfPrep["X_address"].isin(random_elements)
            ].reset_index(drop=True)
            test_data: pd.DataFrame = dfPrep[
                ~dfPrep["X_address"].isin(random_elements)
            ].reset_index(drop=True)
        else:
            # Fallback: split theo hàng khi chỉ có 1 địa chỉ
            split_idx = max(1, int(len(dfPrep) * 0.9)) if len(dfPrep) > 1 else 1
            self.logger.info(
                f"Row-wise split because only {n_addrs} address(es). split_idx={split_idx}"
            )
            train_data: pd.DataFrame = dfPrep.iloc[:split_idx].reset_index(drop=True)
            test_data: pd.DataFrame = dfPrep.iloc[split_idx:].reset_index(drop=True)

        # Các cột bỏ đi (thêm Label_bool để tránh leak target)
        drop_cols: List[str] = [
            "_id",
            "SubX_Time",
            "SubX_From_time",
            "SubX_To_time",
            "X_From_time",
            "X_To_time",
            "X_Time",
            "X_address",
            "SubX_address",
            "Label_bool",
        ]
        if getattr(self, "compute_embedding_similarity", False):
            drop_cols += ["X_Diff2VecEmbedding", "SubX_Diff2VecEmbedding"]

        # Xóa cột chỉ khi tồn tại
        train_data.drop(
            columns=[c for c in drop_cols if c in train_data.columns],
            inplace=True,
            errors="ignore",
        )
        test_data.drop(
            columns=[c for c in drop_cols if c in test_data.columns],
            inplace=True,
            errors="ignore",
        )

        # Log sau khi drop cột + 5 dòng đầu
        self.logger.info(
            f"Train data shape: {train_data.shape}; cols: {list(train_data.columns)}"
        )
        self.logger.info(
            f"Test  data shape: {test_data.shape}; cols: {list(test_data.columns)}"
        )
        self.logger.info(f"Train head:\n{train_data.head(5).to_string(index=False)}")
        self.logger.info(f"Test  head:\n{test_data.head(5).to_string(index=False)}")

        # Cảnh báo nếu rỗng
        if train_data.empty:
            self.logger.warning("Train data is EMPTY after split/filter.")
        if test_data.empty:
            self.logger.warning("Test data is EMPTY after split/filter.")

        # Lưu file CSV
        train_data.to_csv(f"{self.saving_path}/train_data.csv", index=False)
        test_data.to_csv(f"{self.saving_path}/test_data.csv", index=False)

        self.logger.info("Successful")
