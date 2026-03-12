from __future__ import annotations

from collections.abc import Sequence

from fastapi import HTTPException, UploadFile, status


SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/bmp"})


def validate_image_upload(file: UploadFile) -> None:
    if file.content_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Upload a JPEG, PNG, BMP, or WEBP image.",
        )


def validate_image_uploads(files: Sequence[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    invalid_files = [file.filename or "unknown" for file in files if file.content_type not in SUPPORTED_IMAGE_MEDIA_TYPES]
    if invalid_files:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file types for: {', '.join(invalid_files)}",
        )