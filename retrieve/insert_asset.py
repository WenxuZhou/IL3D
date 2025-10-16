from qdrant_3d_client import QdrantMultiVectorFor3D
from text_embedding import TextEmbeddingModel
from qdrant_client import models


import json
import os


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def insert_asset(
    data_json, text_embedding_model: TextEmbeddingModel, client: QdrantMultiVectorFor3D
):
    """
    Args:
        data_json: list[dict]

    Example:
    {
        {
        "dataset": "3D-FRONT",
        "model_id": "f89da2db-ad8c-4582-b186-ed2a46f3cb15",
        "category": "armchair",
        "meta_data": {
            "category": "chair",
            "synset": "chair.n.01",
            "width": 0.641075998544693,
            "length": 0.9789590239524841,
            "height": 0.7714189887046814,
            "volume": 0.4841326320913875,
            "mass": 5,
            "frontview": 0,
            "description": "A modern armchair with a tufted backrest and metal legs.",
            "materials": [
                "fabric",
                "metal"
            ],
            "onCeiling": false,
            "onWall": false,
            "onFloor": true,
            "onObject": false,
            "scale": [
                1.222649450397238,
                1.0,
                1.0656980473776703
            ]
        },
        "label": "Armchairs",
        "path": "3D-FRONT/Armchairs/f89da2db-ad8c-4582-b186-ed2a46f3cb15.usdz"
        }    
    }
    """

    for idx, item in enumerate(data_json[:]):
        payload = {
            "description": item["meta_data"]["description"],
            "category": item["category"],
            "frontview": item["meta_data"]["frontview"],
            "path": item["path"],
            "dataset": item["dataset"],
            "label": item["label"],
            "onCeiling": item["meta_data"]["onCeiling"],
            "onWall": item["meta_data"]["onWall"],
            "onFloor": item["meta_data"]["onFloor"],
            "onObject": item["meta_data"]["onObject"],
            "scale":(item["meta_data"]["scale"][0], item["meta_data"]["scale"][1], item["meta_data"]["scale"][2]),
            "boundingbox":(item["meta_data"]["width"], item["meta_data"]["length"], item["meta_data"]["height"])
        }

        try:
            category_string = payload["label"] + "," + payload["category"]
        except:
            category_string = payload["label"]

        description_embedding = text_embedding_model.text_embedding([category_string])[
            0
        ].tolist()
        category_embedding = text_embedding_model.text_embedding([category_string])[
            0
        ].tolist()

        client.upsert(
            points=[
                models.PointStruct(
                    id=idx,
                    payload=payload,
                    vector={
                        "text_description": description_embedding,
                        "text_category": category_embedding,
                    },
                )
            ]
        )

        print(f"inserted {idx}/{len(data_json)}")


if __name__ == "__main__":
    client = QdrantMultiVectorFor3D(client_path="data/qdrant")
    text_embedding_model = TextEmbeddingModel(cache_dir="data/text_emb")

    # 读取json文件
    root_path = "data/"
    data_json_path = os.path.join(root_path, "assets.json")
    data_json = load_json_file(data_json_path)
    insert_asset(data_json, text_embedding_model, client)
    # for item in data_json:
    #     if 'obj' in item["dataset"]:
    #         print(item)
    #         break 
           
   