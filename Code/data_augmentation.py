# Data Augmentation
import os
import cv2
import albumentations as A
import random
import numpy as np

main_folder = "New Volume/HABSWatch_extension/Data_Augmentation/2026 Chaetoceros paper_revisions/Base_Training set"
valid_ext = (".jpg", ".jpeg", ".png")

def get_transform_with_name(image):

    # Flips
    hflip = A.HorizontalFlip(p=1.0)
    vflip = A.VerticalFlip(p=1.0)

    # Rotation

    rotate_neg = A.Rotate(limit=(-20, -5), border_mode=cv2.BORDER_REPLICATE, p=1.0)
    rotate_pos = A.Rotate(limit=(5, 20), border_mode=cv2.BORDER_REPLICATE, p=1.0)

    rotate = A.OneOf([rotate_neg, rotate_pos], p=1.0)

    # Scaling 
    zoom_out = A.Affine(
        scale=(0.8, 0.95),
        translate_percent=0,
        rotate=0,
        shear=0,
        fit_output=False,
        border_mode=cv2.BORDER_REPLICATE,
        p=1.0
    )

    zoom_in = A.Affine(
        scale=(1.05, 1.2),
        translate_percent=0,
        rotate=0,
        shear=0,
        fit_output=False,
        border_mode=cv2.BORDER_REPLICATE,
        p=1.0
    )

    zoom = A.OneOf([zoom_out, zoom_in], p=1.0)

    transformations = [
        ("hflip", [hflip]),
        ("hflip_scale", [hflip, zoom]),
        ("hflip_rotate", [hflip, rotate]),
        ("vflip", [vflip]),
        ("vflip_scale", [vflip, zoom]),
        ("vflip_rotate", [vflip, rotate]),
        ("rotate", [rotate]),
        ("scale", [zoom]),
    ]

    name, transform_list = random.choice(transformations)

    return A.Compose(transform_list), name


for root, dirs, files in os.walk(main_folder):

    for file in files:

        if file.lower().endswith(valid_ext):

            image_path = os.path.join(root, file)

            #preserves alpha channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                continue

            transform, name = get_transform_with_name(image)

            augmented = transform(image=image)
            aug_image = augmented["image"]

            print(f"Augmenting: {file}")

            base_name = os.path.splitext(file)[0]
            new_name = f"{base_name}_{name}.png"

            save_folder = os.path.join(root, "augmented")
            os.makedirs(save_folder, exist_ok=True)

            save_path = os.path.join(save_folder, new_name)
            cv2.imwrite(save_path, aug_image)

print("Augmentation complete!")