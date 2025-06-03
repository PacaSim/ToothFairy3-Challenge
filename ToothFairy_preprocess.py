import os
import json
import shutil
import nibabel as nib
import numpy as np
from collections import defaultdict

def strip_image_id(filename):
    """ToothFairy3F_001_0000 → ToothFairy3F_001"""
    return "_".join(filename.replace(".nii.gz", "").split("_")[:2])

def collect_image_files_by_prefix(images_dir, prefixes):
    grouped = defaultdict(list)
    for fname in os.listdir(images_dir):
        if fname.endswith(".nii.gz"):
            for prefix in prefixes:
                if fname.startswith(prefix):
                    grouped[prefix].append(fname)
                    break
    return grouped

def collect_label_set(labels_dir):
    return set(fname.replace(".nii.gz", "") for fname in os.listdir(labels_dir) if fname.endswith(".nii.gz"))

def move_matched_pairs(images_dir, labels_dir, output_images_dir, output_labels_dir, prefixes, max_per_prefix=17):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_groups = collect_image_files_by_prefix(images_dir, prefixes)
    label_set = collect_label_set(labels_dir)

    for prefix in prefixes:
        matched = []
        for fname in sorted(image_groups[prefix], reverse=True):
            base_id = strip_image_id(fname)
            label_fname = base_id + ".nii.gz"
            if base_id in label_set:
                matched.append((fname, label_fname))
            if len(matched) >= max_per_prefix:
                break

        print(f"[{prefix}] Matching pairs found: {len(matched)}")

        for image_fname, label_fname in matched:
            shutil.move(os.path.join(images_dir, image_fname), os.path.join(output_images_dir, image_fname))
            shutil.move(os.path.join(labels_dir, label_fname), os.path.join(output_labels_dir, label_fname))

def remap_label_ids_from_json(json_path, save_json_path, num_training):
    with open(json_path, 'r') as f:
        data = json.load(f)

    original_labels = data["labels"]
    label_items = sorted(original_labels.items(), key=lambda x: int(x[1]))
    new_id_map = {int(old_id): new_id for new_id, (_, old_id) in enumerate(label_items)}

    remapped_labels = {name: new_id for new_id, (name, _) in enumerate(label_items)}


    output_dict = {
        "labels": remapped_labels,
        "numTraining": num_training + 1,
        "numTest": 51,
        "file_ending": ".nii.gz",
        "channel_names": {
            "0": "CBCT"
        }
    }

    with open(save_json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)

    return new_id_map

def remap_labels_in_nii(input_path, output_path, id_mapping):
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.uint16)

    remapped_data = np.zeros_like(data, dtype=np.uint16)
    for old_id, new_id in id_mapping.items():
        remapped_data[data == old_id] = new_id

    new_img = nib.Nifti1Image(remapped_data, affine=img.affine, header=img.header)
    nib.save(new_img, output_path)

def remap_all_labels(label_dir, output_dir, id_mapping):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(label_dir):
        if fname.endswith(".nii.gz"):
            input_path = os.path.join(label_dir, fname)
            output_path = os.path.join(output_dir, fname)
            remap_labels_in_nii(input_path, output_path, id_mapping)


def main(input_root, output_root):
    input_images_dir = os.path.join(input_root, "imagesTr")
    input_labels_dir = os.path.join(input_root, "labelsTr")
    input_json_path = os.path.join(input_root, "dataset.json")

    output_imagesTr_dir = os.path.join(output_root, "imagesTr")
    output_labelsTr_dir = os.path.join(output_root, "labelsTr")
    output_imagesVal_dir = os.path.join(output_root, "imagesVal")
    output_labelsVal_dir = os.path.join(output_root, "labelsVal")
    output_json_path = os.path.join(output_root, "dataset.json")

    prefixes = ["ToothFairy3F", "ToothFairy3P", "ToothFairy3S"]

    print("Remapping label IDs and writing new dataset.json...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    id_mapping = remap_label_ids_from_json(input_json_path, output_json_path, num_training=0)  # num_training은 나중에 수정

    print("Remapping labels and copying images...")
    os.makedirs(output_labelsTr_dir, exist_ok=True)
    remap_all_labels(input_labels_dir, output_labelsTr_dir, id_mapping)
    shutil.copytree(input_images_dir, output_imagesTr_dir, dirs_exist_ok=True)

    print("Moving matched image-label pairs to validation set...")
    move_matched_pairs(output_imagesTr_dir, output_labelsTr_dir, output_imagesVal_dir, output_labelsVal_dir, prefixes, max_per_prefix=17)

    total_labels = [f for f in os.listdir(output_labelsTr_dir) if f.endswith(".nii.gz")]
    num_val = len([f for f in os.listdir(output_labelsVal_dir) if f.endswith(".nii.gz")])
    num_train = len(total_labels) - num_val

    print("Updating dataset.json with actual numTraining...")
    with open(output_json_path, 'r') as f:
        dataset = json.load(f)
    dataset["numTraining"] = num_train
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print("Done.")



if __name__ == "__main__":
    input_root = "dataset/ToothFairy3"
    output_root = "dataset/arranged_toothFairy3"
    main(input_root, output_root)
