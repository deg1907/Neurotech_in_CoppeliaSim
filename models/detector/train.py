"""
Обучение CNN-детектора промышленных деталей.

Запуск:
  python train.py --dataset ../../../../dataset --epochs 50 --batch 16

Структура датасета (YOLO формат):
  dataset/
    images/train/  val/  test/
    labels/train/  val/  test/   ← class x_c y_c w h (нормализованные)
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Путь к модулям проекта
sys.path.insert(0, str(Path(__file__).parent))
from architecture import Detector
from loss import DetectionLoss


class PartDataset(Dataset):
    """
    Датасет промышленных деталей в формате YOLO.

    Каждая пара (изображение, метка):
      - изображение: (3, 640, 640) float32, нормализованное [0, 1]
      - метка: dict {'class': int, 'bbox': [x_c, y_c, w, h]}

    Args:
        img_dir:  папка с изображениями (*.png)
        lbl_dir:  папка с метками (*.txt, YOLO формат)
        img_size: размер стороны после ресайза
    """

    def __init__(self, img_dir: str, lbl_dir: str, img_size: int = 640) -> None:
        self.img_size = img_size
        self.img_paths = sorted(Path(img_dir).glob('*.png'))
        self.lbl_dir   = Path(lbl_dir)

        # Оставляем только файлы, для которых есть метка
        self.img_paths = [
            p for p in self.img_paths
            if (self.lbl_dir / (p.stem + '.txt')).exists()
        ]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')

        # Загрузка и препроцессинг изображения
        img_bgr = cv2.imread(str(img_path))
        img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t   = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        img_t   = img_t.permute(2, 0, 1)  # (H,W,3) → (3,H,W)

        # Загрузка метки (первая строка файла)
        with open(lbl_path) as f:
            line = f.readline().strip().split()
        cls = int(line[0])
        x_c, y_c, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])

        target = {'class': cls, 'bbox': [x_c, y_c, w, h]}
        return img_t, target


def collate_fn(batch: list) -> tuple:
    """Объединяет список (image, target) в батч."""
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def train(config: dict) -> None:
    """
    Обучение детектора.

    Args:
        config: словарь с ключами:
          dataset_root, epochs, batch_size, lr, device, save_dir, num_classes
    """
    dataset_root = Path(config['dataset_root'])
    save_dir     = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config['device'])
    print(f'Устройство: {device}')

    # ── Датасеты и загрузчики ────────────────────────────────────────────────
    train_ds = PartDataset(
        img_dir=str(dataset_root / 'images' / 'train'),
        lbl_dir=str(dataset_root / 'labels' / 'train'),
    )
    print(train_ds)  # Проверка загрузки
    val_ds = PartDataset(
        img_dir=str(dataset_root / 'images' / 'val'),
        lbl_dir=str(dataset_root / 'labels' / 'val'),
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            f'Train датасет пуст: {dataset_root / "images" / "train"}\n'
            f'Сначала сгенерируй датасет:\n'
            f'  cd dataset/generation\n'
            f'  python generate_dataset.py --n 3000 --dr full --split'
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f'Val датасет пуст: {dataset_root / "images" / "val"}\n'
            f'Запусти генерацию с флагом --split'
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    print(f'Train: {len(train_ds)} изображений, Val: {len(val_ds)} изображений')

    # ── Модель, loss, оптимизатор ────────────────────────────────────────────
    model = Detector(num_classes=config['num_classes']).to(device)
    criterion = DetectionLoss(num_classes=config['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── Логирование в CSV ────────────────────────────────────────────────────
    log_path = save_dir / 'train_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    best_val_loss = float('inf')

    # ── Цикл обучения ────────────────────────────────────────────────────────
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss_sum = 0.0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)

            optimizer.zero_grad()
            preds = model(imgs)                          # (B, 1+4+C, 20, 20)
            loss, components = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss_sum += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f'Эпоха {epoch}/{config["epochs"]} '
                    f'[{batch_idx+1}/{len(train_loader)}]  '
                    f'loss={loss.item():.4f}  '
                    f'obj={components["obj"]:.3f}  '
                    f'bbox={components["bbox"]:.3f}  '
                    f'cls={components["cls"]:.3f}'
                )

        # ── Валидация ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                val_loss_sum += loss.item()

        train_loss_avg = train_loss_sum / len(train_loader)
        val_loss_avg   = val_loss_sum   / len(val_loader)
        current_lr     = optimizer.param_groups[0]['lr']

        print(
            f'[Эпоха {epoch}] train_loss={train_loss_avg:.4f}  '
            f'val_loss={val_loss_avg:.4f}  lr={current_lr:.2e}'
        )

        # Логируем в CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss_avg, val_loss_avg, current_lr])

        # Сохраняем лучшую модель
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            ckpt_path = save_dir / 'detector_best.pt'
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': val_loss_avg}, ckpt_path)
            print(f'  → Сохранён лучший checkpoint: {ckpt_path}')

        scheduler.step(val_loss_avg)

    print(f'Обучение завершено. Лучший val_loss={best_val_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение детектора')
    parser.add_argument('--dataset', type=str, default='../../dataset',
                        help='Путь к корню датасета')
    parser.add_argument('--epochs',  type=int,   default=50)
    parser.add_argument('--batch',   type=int,   default=16)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--device',  type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save',    type=str,   default='../../weights',
                        help='Папка для сохранения весов')
    parser.add_argument('--classes', type=int,   default=3)
    args = parser.parse_args()

    train({
        'dataset_root': args.dataset,
        'epochs':       args.epochs,
        'batch_size':   args.batch,
        'lr':           args.lr,
        'device':       args.device,
        'save_dir':     args.save,
        'num_classes':  args.classes,
    })
