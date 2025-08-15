from constants.network_constants import Chains
from service.query_subgraph import query_subgraph
from utils.embedding_utils import EmbeddingUtils
from utils.logger_utils import get_logger

logger = get_logger("Subgraphs Exporter Scheduler")


def node_embedding_exporter(
    saving_path: str,
    chain: str = "ethereum",
    radius: int = 2,
):
    """Export Diff2Vec embeddings from a subgraph.

    Args:
        chain (str): Network name (e.g. "bsc", "polygon").
        radius (int): Radius of subgraph.
        saving_path (str): Directory to save output CSV file.
    """
    chain: str = chain.lower()
    if chain not in Chains.mapping:
        raise ValueError(f"Chain '{chain}' is not supported")

    subgraph_df = query_subgraph(chain, radius)
    logger.info("Successful query subgraph")

    subgraph_df["Diff2VecEmbedding"] = subgraph_df.apply(
        lambda row: EmbeddingUtils.get_diff2vec_embedding(row), axis=1
    )
    subgraph_df = subgraph_df.explode(["vertices", "Diff2VecEmbedding"])
    subgraph_df = subgraph_df[["_id", "vertices", "Diff2VecEmbedding"]]

    output_path: str = f"{saving_path}/embedding_df.csv"
    subgraph_df.to_csv(output_path, index=False)
    logger.info(f"Embedding saved to {output_path}")
