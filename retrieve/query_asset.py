from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .qdrant_3d_client import QdrantMultiVectorFor3D
from .text_embedding import TextEmbeddingModel

def query_asset(qdrant_client: QdrantMultiVectorFor3D, query_description: list[float], limit: int = 5):
    result = qdrant_client.client.query_points(
        collection_name=qdrant_client.collection_name,
        query=query_description,
        using="text_description",
        limit=limit
    )
    return result


def query_asset_with_filter_on_floor(qdrant_client: QdrantMultiVectorFor3D, query_description: list[float], limit: int = 5, dataset: str = None):
    query_filter = None
    
    query_filter = Filter(
        must=[
            FieldCondition(
                key="dataset",
                match=MatchValue(value=dataset)
            ),
            FieldCondition(
                key="onFloor",
                match=MatchValue(value=True)
            )
        ]
    )
    
    result = qdrant_client.client.query_points(
        collection_name=qdrant_client.collection_name,
        query=query_description,
        using="text_description",
        query_filter=query_filter,
        limit=limit,
    )
    return result

def query_asset_with_filter_on_wall(qdrant_client: QdrantMultiVectorFor3D, query_description: list[float], limit: int = 5, dataset: str = None):
    query_filter = None
    
    query_filter = Filter(
        must=[
            FieldCondition(
                key="dataset",
                match=MatchValue(value=dataset)
            ),
            FieldCondition(
                key="onWall",
                match=MatchValue(value=True)
            )
        ]
    )
    
    result = qdrant_client.client.query_points(
        collection_name=qdrant_client.collection_name,
        query=query_description,
        using="text_description",
        query_filter=query_filter,
        limit=limit,
    )
    return result

if __name__ == "__main__":
    qdrant_client = QdrantMultiVectorFor3D(client_path="data/qdrant")
    text_embedding_model = TextEmbeddingModel(cache_dir="data/text_emb")
    test_query = [
        "a brown sofa"
    ]
   
    for query in test_query:
        query_description = text_embedding_model.text_embedding([query])[0].tolist()
        
        # 不使用过滤器的查询
        result = query_asset(qdrant_client, query_description, limit=5)
        print(f"无过滤器查询结果: {result}")
        print('*'*50)
        
        # 使用 dataset 过滤器的查询（假设有 HSSD 数据集）
        result_filtered = query_asset_with_filter_on_wall(qdrant_client, query_description, limit=5, dataset="HSSD")
        print(f"过滤 HSSD 数据集查询结果: {result_filtered}")
        print('*'*100)

        for item in result_filtered.points:
            print([item.id, item.score,item.payload])
