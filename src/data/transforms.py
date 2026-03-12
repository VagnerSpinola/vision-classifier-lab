from __future__ import annotations

from torchvision import transforms

from app.core.config import AugmentationConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int,
    augmentation: AugmentationConfig | None = None,
) -> transforms.Compose:
    augmentation = augmentation or AugmentationConfig()
    operations: list[transforms.Compose | transforms.RandomHorizontalFlip | transforms.RandomVerticalFlip | transforms.RandomRotation | transforms.ColorJitter | transforms.ToTensor | transforms.Normalize] = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=augmentation.random_resized_crop_scale),
    ]

    if augmentation.horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    if augmentation.vertical_flip:
        operations.append(transforms.RandomVerticalFlip(p=0.2))
    if augmentation.rotation_degrees > 0:
        operations.append(transforms.RandomRotation(augmentation.rotation_degrees))
    if augmentation.color_jitter > 0:
        operations.append(
            transforms.ColorJitter(
                brightness=augmentation.color_jitter,
                contrast=augmentation.color_jitter,
                saturation=augmentation.color_jitter,
            )
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )