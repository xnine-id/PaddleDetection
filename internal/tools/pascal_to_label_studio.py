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

    filename_elem = root.find("filename")
    width_elem = root.find("size/width")
    height_elem = root.find("size/height")
    if filename_elem is None or width_elem is None or height_elem is None:
        continue

    filename = filename_elem.text
    if filename is None:
        continue
    
    width_text = width_elem.text
    height_text = height_elem.text
    if width_text is None or height_text is None:
        continue
    width = int(width_text)
    height = int(height_text)

    results = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bbox = obj.find("bndbox")
        if name_elem is None or bbox is None:
            continue
        label = name_elem.text
        if label is None:
            continue

        xmin_elem = bbox.find("xmin")
        ymin_elem = bbox.find("ymin")
        xmax_elem = bbox.find("xmax")
        ymax_elem = bbox.find("ymax")
        if xmin_elem is None or ymin_elem is None or xmax_elem is None or ymax_elem is None:
            continue

        xmin_text = xmin_elem.text
        ymin_text = ymin_elem.text
        xmax_text = xmax_elem.text
        ymax_text = ymax_elem.text
        if xmin_text is None or ymin_text is None or xmax_text is None or ymax_text is None:
            continue

        xmin = int(xmin_text)
        ymin = int(ymin_text)
        xmax = int(xmax_text)
        ymax = int(ymax_text)

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