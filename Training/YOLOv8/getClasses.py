import os

annotation_dir = "C:/training/kaggle/Bounding Boxes - YOLO Format - 1"

def get_class_ids(annotation_dir):
    class_ids = set()
    for ann_file in os.listdir(annotation_dir):
        if not ann_file.endswith(".txt"):
            continue
        ann_path = os.path.join(annotation_dir, ann_file)
        try:
            with open(ann_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
                    except (IndexError, ValueError):
                        print(f"Invalid line in {ann_file}: {line.strip()}")
        except Exception as e:
            print(f"Error reading {ann_file}: {str(e)}")
    return sorted(class_ids)

class_ids = get_class_ids(annotation_dir)
print(f"Detected class IDs: {class_ids}")
print(f"Number of classes: {len(class_ids)}")