from qdrant_client import QdrantClient, models


class QdrantMultiVectorFor3D:
    def __init__(
        self, client_path: str = "qdrant_db_benckmark", collection_name: str = "assets"
    ):
        self.client = QdrantClient(path=client_path)
        self.collection_name = collection_name

        # check if collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_description": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    ),
                    "text_category": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    )
                }
            )

    def upsert(self, points: list[models.PointStruct]):
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query: list[float], limit: int = 10):
        """
        Args:
            query: list[float]   2*n vector
        Returns:
            list[models.PointStruct]
        """
        result = self.client.query_points(
            collection_name=self.collection_name, query=query, limit=limit
        )
        return result



if __name__ == "__main__":
    qdrant_client_1 = QdrantMultiVectorFor3D()
    # qdrant_client_2 = QdrantMultiVectorFor3D()
    
    # qdrant_client_1.upsert(points=[
    #     models.PointStruct(
    #         id=1,
    #         payload={"text": "1", "image": "1.png"},
    #         vector={
    #             "text_description": [-0.013,  0.020, -0.007, -0.111],
    #             "text_category": [0.2]
    #         }
    #     )
    # ])
    
