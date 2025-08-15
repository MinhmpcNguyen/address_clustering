import re

import numpy as np
import numpy.typing as npt
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from diff2vec.diffusion_2_vec import (
    learn_pooled_embeddings,
    run_parallel_feature_creation,
)


class EmbeddingUtils:
    @staticmethod
    def get_diff2vec_embedding(row):
        """
        Generate Diff2Vec embedding for a given row of data.
        Args:
            row (dict): A dictionary containing 'edges' and 'vertices'.
        Returns:
            list (npt.NDArray[np.float_]): A list of embedding vectors for the vertices.
        Raises:
            ValueError: If 'edges' or 'vertices' are missing or invalid.
        """
        # if "edges" not in row or "vertices" not in row:
        #     raise ValueError("Row must contain 'edges' and 'vertices' keys.")
        # if not isinstance(row["edges"], str) or not isinstance(row["vertices"], list):
        #     raise ValueError("'edges' must be a string and 'vertices' must be a list.")

        try:
            walks, counts = run_parallel_feature_creation(
                edge_list_path=row["edges"], vertex_set_card=16, replicates=4, workers=4
            )
            model: Word2Vec = learn_pooled_embeddings(walks, counts)
            embedding_row: list[npt.NDArray[np.float_]] = list(
                map(lambda x: model.wv.get_vector(x), row["vertices"])
            )
            return embedding_row
        except Exception as e:
            raise RuntimeError(f"Failed to generate Diff2Vec embedding: {e}")

    @staticmethod
    def diff_cosine(row: dict[str, npt.NDArray[np.float_]]) -> float:
        """
        Calculate the cosine similarity between two embeddings of two wallets.
        Args:
            row (dict): A dictionary containing 'X_Diff2VecEmbedding' and 'SubX_Diff2VecEmbedding'.
        Returns:
            float: Cosine similarity between the two embeddings, or 0.0 if invalid.
        """
        if "X_Diff2VecEmbedding" not in row or "SubX_Diff2VecEmbedding" not in row:
            raise ValueError(
                "Row must contain 'X_Diff2VecEmbedding' and 'SubX_Diff2VecEmbedding' keys."
            )

        vec1: npt.NDArray[np.float_] = np.array(row["X_Diff2VecEmbedding"]).reshape(
            1, -1
        )
        vec2: npt.NDArray[np.float_] = np.array(row["SubX_Diff2VecEmbedding"]).reshape(
            1, -1
        )

        try:
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            cosine_sim: float = cosine_similarity(vec1, vec2)[0][0]
            return cosine_sim
        except Exception as e:
            raise RuntimeError(f"Failed to calculate cosine similarity: {e}")

    @staticmethod
    def get_embedding_list(row: str):
        """
        Convert an embedding vector in NodeEmbedding to a list of floats.
        Args:
            row (str): A string representation of the embedding vector.
        Returns:
            list: A list of floats representing the embedding vector.
        """

        numbers: list[str] = re.findall(r"-?\d+\.\d+", row)
        if not numbers:
            raise ValueError("No valid numbers found in the input string.")

        number_list: list[float] = [float(number) for number in numbers]
        return number_list
