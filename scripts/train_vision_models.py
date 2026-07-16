#!/usr/bin/env python3
"""Train board cell classifier and inventory slot masker; export ONNX weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from block_blast_solver import config
from block_blast_solver.modules import vision
from block_blast_solver.modules.vision_fixtures import iter_skin_fixtures
from block_blast_solver.modules.vision_models import ModelRegistry


class BoardCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlotUNetLite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class BoardCellDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.samples: list[tuple[Path, float]] = []
        for path in sorted((root / "occupied").glob("*.png")):
            self.samples.append((path, 1.0))
        for path in sorted((root / "empty").glob("*.png")):
            self.samples.append((path, 0.0))
        if not self.samples:
            raise SystemExit(f"No board cell samples under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read {path}")
        tensor = _bgr_to_nchw(image, (32, 32))
        return tensor, torch.tensor([label], dtype=torch.float32)


class InventoryMaskDataset(Dataset):
    def __init__(self, root: Path) -> None:
        image_dir = root / "images"
        mask_dir = root / "masks"
        self.pairs = []
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise SystemExit(f"Missing mask for {image_path.name}")
            self.pairs.append((image_path, mask_path))
        if not self.pairs:
            raise SystemExit(f"No inventory pairs under {root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"failed to read inventory pair {image_path.name}")
        image_tensor = _bgr_to_nchw(image, (128, 128))
        mask_resized = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))[None, ...]
        return image_tensor, mask_tensor


def _bgr_to_nchw(image_bgr: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


def _train_classifier(data_root: Path, epochs: int, device: torch.device) -> BoardCNN:
    dataset = BoardCellDataset(data_root / "board_cells")
    occupied = sum(1 for _, label in dataset.samples if label >= 0.5)
    empty = len(dataset) - occupied
    pos_weight = torch.tensor([empty / max(1, occupied)], dtype=torch.float32, device=device)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    model = BoardCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_x.size(0)
        print(f"board epoch {epoch + 1}/{epochs} loss={total / len(dataset):.4f}")
    return model


def _train_masker(data_root: Path, epochs: int, device: torch.device) -> SlotUNetLite:
    dataset = InventoryMaskDataset(data_root / "inventory")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = SlotUNetLite().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_x.size(0)
        print(f"mask epoch {epoch + 1}/{epochs} loss={total / len(dataset):.4f}")
    return model


class _ProbBoardWrapper(nn.Module):
    def __init__(self, model: BoardCNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class _ProbMaskWrapper(nn.Module):
    def __init__(self, model: SlotUNetLite) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def _export_onnx(model: nn.Module, path: Path, input_shape: tuple[int, ...]) -> None:
    model.eval()
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["prob"],
        opset_version=17,
        dynamo=False,
    )


def _evaluate_fixtures() -> list[str]:
    ModelRegistry.reset_for_tests()
    registry = ModelRegistry.get()
    if not registry.using_learned:
        return ["models_not_loaded"]

    failures: list[str] = []
    for fixture in iter_skin_fixtures():
        previous_board = config.BOARD_ROI
        previous_pieces = config.PIECES_ROI
        try:
            config.BOARD_ROI = fixture.board_roi
            config.PIECES_ROI = fixture.pieces_roi
            board, occluded = vision.get_board_state(fixture.image)
            if occluded or board.tolist() != fixture.board.tolist():
                failures.append(f"{fixture.name}:board")
                continue
            cell_w = fixture.image.shape[1] * fixture.board_roi[2] / 8.0
            cell_h = fixture.image.shape[0] * fixture.board_roi[3] / 8.0
            pieces = vision.get_pieces(fixture.image, cell_w, cell_h)
            for got, expected in zip(pieces, fixture.pieces):
                if expected is None:
                    if got is not None:
                        failures.append(f"{fixture.name}:pieces")
                        break
                elif got is None or got.tolist() != expected.tolist():
                    failures.append(f"{fixture.name}:pieces")
                    break
            else:
                print(f"PASS {fixture.name}")
                continue
            print(f"FAIL {fixture.name}")
        finally:
            config.BOARD_ROI = previous_board
            config.PIECES_ROI = previous_pieces
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--board-epochs", type=int, default=40)
    parser.add_argument("--mask-epochs", type=int, default=120)
    args = parser.parse_args()

    device = torch.device("cpu")
    board_model = _train_classifier(args.data, args.board_epochs, device)
    mask_model = _train_masker(args.data, args.mask_epochs, device)

    board_path = args.out_dir / "board_cell_classifier.onnx"
    mask_path = args.out_dir / "inventory_slot_masker.onnx"
    staging = args.out_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged_board = staging / board_path.name
    staged_mask = staging / mask_path.name

    _export_onnx(_ProbBoardWrapper(board_model.cpu()).eval(), staged_board, (1, 3, 32, 32))
    _export_onnx(_ProbMaskWrapper(mask_model.cpu()).eval(), staged_mask, (1, 3, 128, 128))

    previous_board_path = config.BOARD_CELL_MODEL_PATH
    previous_mask_path = config.INVENTORY_MASK_MODEL_PATH
    try:
        config.BOARD_CELL_MODEL_PATH = str(staged_board)
        config.INVENTORY_MASK_MODEL_PATH = str(staged_mask)
        import block_blast_solver.modules.vision_models as vision_models

        vision_models.DEFAULT_BOARD_MODEL = str(staged_board)
        vision_models.DEFAULT_MASK_MODEL = str(staged_mask)
        failures = _evaluate_fixtures()
    finally:
        config.BOARD_CELL_MODEL_PATH = previous_board_path
        config.INVENTORY_MASK_MODEL_PATH = previous_mask_path
        vision_models.DEFAULT_BOARD_MODEL = previous_board_path
        vision_models.DEFAULT_MASK_MODEL = previous_mask_path
        ModelRegistry.reset_for_tests()

    if failures:
        print("Acceptance gate failed:", ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)

    staged_board.replace(board_path)
    staged_mask.replace(mask_path)
    print(f"Wrote {board_path}")
    print(f"Wrote {mask_path}")


if __name__ == "__main__":
    main()
