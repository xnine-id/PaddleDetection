import os
import json

def main():
    # Paths
    image_dir = "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/frames/vehicle_plate/images"
    gt_path = "/home/jeremyjfn/app/skyure-df/PaddleDetection/storage/private/eval/results/vehicle_plate/ground_truth.json"

    # Load ground truth annotations
    if not os.path.exists(gt_path):
        print(f"Error: Format file not found at {gt_path}")
        return

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} annotations from ground truth.")

    # Extract all annotated image file names
    ground_truth_images = set()
    for item in gt_data:
        image_path_in_json = item.get("image", "")
        # The path looks like: /data/local-files/?d=frames/vehicle_plate/images/Cars199.png
        # We can extract just the filename (the part after the last '/')
        filename = image_path_in_json.split("/")[-1]
        ground_truth_images.add(filename)
    
    print(f"Found {len(ground_truth_images)} unique annotated images in ground truth.")

    # Iterate through image directory and delete non-annotated images
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    all_images = os.listdir(image_dir)
    print(f"Found {len(all_images)} total files in the image directory.")

    deleted_count = 0
    for image_name in all_images:
        # Check if the file is an image based on common extensions
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        if image_name not in ground_truth_images:
            img_path = os.path.join(image_dir, image_name)
            try:
                os.remove(img_path)
                deleted_count += 1
                # print(f"Deleted: {image_name}") # Uncomment to see what is being deleted
            except Exception as e:
                print(f"Failed to delete {image_name}: {e}")
                
    print(f"\nDone. Successfully deleted {deleted_count} unannotated images.")

if __name__ == '__main__':
    main()
