"""
File to submit to leaderboard. Contains preprocessing code for l
oading images and labels from a CSV file, and preparing them for inference.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

IMAGE_SIZE = 224

inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

def determine_column(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"None of {candidates} in CSV columns {columns}")


def determine_image_path(csv_path: Path, file_name: str) -> Path:
    image_path = Path(file_name)
    if image_path.is_absolute():
        return image_path
    return csv_path.parent / image_path


def load_image(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return inference_transform(image)

"""
Args: path: Path to a metadata CSV with at least file_name, Latitude, Longitude.
Returns:
    X: list of normalized image tensors with shape [3, 224, 224].
    y: raw latitude/longitude labels as a float32 tensor with shape [N, 2].
"""
def prepare_data(path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    csv_path = Path(path)
    data_df = pd.read_csv(csv_path)

    columns = data_df.columns.tolist()
    file_col = determine_column(columns, ["file_name", "filename", "image", "path"])
    lat_col = determine_column(columns, ["Latitude", "latitude", "lat"])
    lon_col = determine_column(columns, ["Longitude", "longitude", "lon"])
    X: List[torch.Tensor] = []
    y_rows: list[list[float]] = []

    for _, row in data_df.iterrows():
        image_path = determine_image_path(csv_path, str(row[file_col]))
        X.append(load_image(image_path))
        y_rows.append([float(row[lat_col]), float(row[lon_col])])

    y = torch.tensor(y_rows, dtype=torch.float32)
    return X, y
