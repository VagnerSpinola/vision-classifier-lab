from io import BytesIO

from PIL import Image

from app.inference.preprocess import load_image_from_bytes, preprocess_image


def test_preprocess_image_returns_batched_tensor() -> None:
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    loaded = load_image_from_bytes(buffer.getvalue())
    tensor = preprocess_image(loaded, image_size=224)

    assert loaded.mode == "RGB"
    assert tuple(tensor.shape) == (1, 3, 224, 224)