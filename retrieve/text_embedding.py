from fastembed import TextEmbedding
import numpy as np


class TextEmbeddingModel:
    def __init__(self, cache_dir: str):
        self.embedding_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=cache_dir,
        )

    def text_embedding(self, texts: list[str]) -> list[np.ndarray]:
        """
        Args:
            texts: list[str]
            cache_dir: str

        Returns:
            embeddings: list[np.ndarray]
        """

        embeddings = list(self.embedding_model.embed(texts))
        return embeddings


if __name__ == "__main__":
    texts = [
        "a photo of a cat",
    ]
    cache_dir = "/home/user/workspace/AutoScene/ckpts/bge"
    embeddings = TextEmbeddingModel(cache_dir=cache_dir)

    for i in range(10):
        print(embeddings.text_embedding(texts)[0].tolist())