import json
with open("polygons.json", "r") as file:
    data = json.load(file)

migrated_polygons = []

for polygon in data["polygons"]:
    migrated_polygon = {
        **polygon["attributes"],
        "type": "polygon",
        "tags": ["parking", "places", "transit"],
        "geometry": polygon["geometry"]
    }
    migrated_polygons.append(migrated_polygon)


print(json.dumps(migrated_polygons, indent=4))

with open("migrated_polygons.json", "w") as outfile:
    json.dump(migrated_polygons, outfile, indent=4)
