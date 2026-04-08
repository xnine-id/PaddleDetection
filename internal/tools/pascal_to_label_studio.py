import os
import xml.etree.ElementTree as ET
import json

annotations_dir = "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/frames/vehicle_plate/annotations2"
images_dir = "images"

tasks = []

for file in os.listdir(annotations_dir):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotations_dir, file))
    root = tree.getroot()

    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    results = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": xmin / width * 100,
                "y": ymin / height * 100,
                "width": (xmax - xmin) / width * 100,
                "height": (ymax - ymin) / height * 100,
                "rectanglelabels": [label]
            }
        })

    tasks.append({
        "data": {
            "image": f"/data/local-files/?d=images/{filename}"
        },
        "annotations": [{
            "result": results
        }]
    })

with open("import.json", "w") as f:
    json.dump(tasks, f, indent=2)