import os
from meta_data import read_json_file, save_json_file


category_mapping = {
    "Bedroom": "BedRoom",
    "LivingDiningRoom": "LivingDiningRoom",
    "LivingRoom": "LivingRoom",
    "Library": "Library",
    "KidsRoom": "KidsRoom",
    "DiningRoom": "DiningRoom",
    "Hallway": "OtherRoom",
    "CloakRoom": "CloakRoom",
    "ElderlyRoom": "ElderlyRoom",
    "OtherRoom": "OtherRoom",
    "Kitchen": "Kitchen",
    "Aisle": "Aisle",
    "Bathroom": "BathRoom",
    "StorageRoom": "StorageRoom",
    "OfficeRoom": "OfficeRoom",
    "Stairwell": "OtherRoom",
    "OutdoorSpace": "Balcony",
    "Terrace": "Balcony",
    "NannyRoom": "OtherRoom",
    "Garage": "Garage",
    "LaundryRoom": "LaundryRoom",
    "RecreationRoom": "RecreationRoom",
    "Auditorium": "RecreationRoom",
    "Courtyard": "Balcony",
    "Balcony": "Balcony",
    "EquipmentRoom": "OtherRoom",
    "ClassRoom": "OfficeRoom",
    "BarArea": "OtherRoom"
}


path = "data/layout"
rooms = os.listdir(path)
for room in rooms:
    file = os.path.join(path, room)
    data = read_json_file(file)
    objs = data["objects"]
    key = objs[0]["roomId"]
    value = category_mapping[key]
    for i in range(len(objs)):
        data["objects"][i]["roomId"] = value
    data["meshes"][0]["roomId"] = value
    save_json_file(data, os.path.join("layout", room))
