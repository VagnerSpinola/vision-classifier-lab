from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)


SUPPORTED_ARCHITECTURES = {"resnet18", "efficientnet_b0", "mobilenet_v3_small"}


@dataclass(slots=True)
class ModelBundle:
    model: nn.Module
    feature_dim: int


def build_classifier(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    bundle = _build_backbone(architecture=architecture, pretrained=pretrained)
    model = bundle.model

    if architecture == "resnet18":
        model.fc = nn.Linear(bundle.feature_dim, num_classes)
    elif architecture == "efficientnet_b0":
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(bundle.feature_dim, num_classes),
        )
    elif architecture == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(bundle.feature_dim, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if freeze_backbone:
        _freeze_feature_extractor(model=model, architecture=architecture)

    return model


def _build_backbone(architecture: str, pretrained: bool) -> ModelBundle:
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        return ModelBundle(model=model, feature_dim=model.fc.in_features)

    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        return ModelBundle(model=model, feature_dim=model.classifier[1].in_features)

    if architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        return ModelBundle(model=model, feature_dim=model.classifier[3].in_features)

    raise ValueError(
        f"Unsupported architecture '{architecture}'. Supported values: {sorted(SUPPORTED_ARCHITECTURES)}"
    )


def _freeze_feature_extractor(model: nn.Module, architecture: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if architecture == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if architecture == "efficientnet_b0":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if architecture == "mobilenet_v3_small":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported architecture: {architecture}")


def checkpoint_payload(
    model: nn.Module,
    architecture: str,
    class_names: list[str],
    image_size: int,
    model_version: str = "latest",
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": image_size,
        "model_version": model_version,
        "state_dict": model.state_dict(),
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = str(checkpoint["architecture"])
    num_classes = int(checkpoint["num_classes"])
    model = build_classifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def get_gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return model.layer4[-1]
    if architecture == "efficientnet_b0":
        return model.features[-1]
    if architecture == "mobilenet_v3_small":
        return model.features[-1]
    raise ValueError(f"Unsupported architecture: {architecture}")