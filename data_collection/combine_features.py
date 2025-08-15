import pandas as pd

from service.combine_features_service import (
    ProcessTrainingDataset,
    merge_embedding,
    merge_from_to,
    merge_pair_labels,
)
from utils.logger_utils import get_logger


def combine_features(
    from_path: str,
    to_path: str,
    embedding_path: str,
    pair_path: str,
    saving_path: str,
    max_workers: int,
    batch_size: int,
    compute_embedding_similarity: bool = False,
):
    """Combine from/to/embedding features and generate final training dataset.

    Args:
        from_path (str): Path to `from_df.csv`.
        to_path (str): Path to `to_df.csv`.
        embedding_path (str): Path to `embedding_df.csv`.
        pair_path (str): Path to `pair.csv` (wallet pairs).
        saving_path (str): Path to save processed training dataset.
        max_workers (int): Number of workers for parallel processing.
        batch_size (int): Size of each processing batch.
    """
    logger = get_logger("CombineFeatures")

    df_from: pd.DataFrame = pd.read_csv(from_path)
    df_to: pd.DataFrame = pd.read_csv(to_path)
    depo_embedding = pd.read_csv(embedding_path)
    depo_embedding.reset_index(drop=True, inplace=True)

    depo_df = merge_from_to(df_from, df_to)
    logger.info("Profile wallet successful")
    if compute_embedding_similarity:
        logger.info("Computing embedding similarity ...")
        depo_df = merge_embedding(depo_df, depo_embedding)
    else:
        logger.info("Skipping embedding similarity computation.")
    depo_pairs = pd.read_csv(pair_path)
    logger.info("Generating training dataset by adding labels ...")

    training_df = merge_pair_labels(
        depo_df,
        depo_pairs,
    )
    logger.info("Successful generate training dataset")

    training_index = list(training_df.index)
    logger.info("Generating time columns and embedding similarity ...")

    job = ProcessTrainingDataset(
        training_dataset=training_df,
        list_index=training_index,
        logger=logger,
        saving_path=saving_path,
        max_workers=max_workers,
        batch_size=batch_size,
        compute_embedding_similarity=compute_embedding_similarity,
    )
    job.run()
