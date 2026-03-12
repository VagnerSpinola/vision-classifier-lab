import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.loss import build_loss
from src.training.trainer import Trainer


def test_build_loss_accepts_class_weights() -> None:
    weights = torch.tensor([1.0, 2.0])
    criterion = build_loss(class_weights=weights, label_smoothing=0.1)

    assert criterion.weight is not None
    assert torch.equal(criterion.weight.cpu(), weights)


def test_trainer_evaluate_returns_test_metrics(tmp_path) -> None:
    model = nn.Linear(4, 2)
    dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    loader = DataLoader(dataset, batch_size=4)
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        class_names=["class_a", "class_b"],
        checkpoint_path=tmp_path / "model.pt",
        registry_path=tmp_path / "registry.pt",
        architecture="resnet18",
        image_size=224,
        best_metric="f1_score",
        early_stopping_patience=2,
        tracker=None,
    )

    metrics = trainer.evaluate(loader, split_name="test")

    assert "test_accuracy" in metrics
    assert "test_f1_score" in metrics