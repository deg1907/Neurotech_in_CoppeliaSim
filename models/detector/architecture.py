"""
Архитектура кастомного CNN-детектора промышленных деталей.

Одностадийный детектор на основе сетки 20×20.
Каждая ячейка предсказывает:
  - objectness: вероятность наличия объекта (1 значение)
  - bbox:        tx, ty — смещение центра внутри ячейки; tw, th — размеры
  - class:       логиты классов (C значений)

Декодирование bbox из ячейки (gi=строка, gj=столбец), G=20:
  x_c = (gj + sigmoid(tx)) / G
  y_c = (gi + sigmoid(ty)) / G
  w   = sigmoid(tw)
  h   = sigmoid(th)
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Базовый строительный блок: Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Остаточный блок: два слоя ConvBNReLU + skip connection.

    Число каналов не меняется — сложение без проекции.
    Input/Output: (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels, 3, 1, 1),
            ConvBNReLU(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # residual connection


class Backbone(nn.Module):
    """
    Backbone: последовательное уменьшение разрешения stride=2 свёртками.

    Input:  (B,   3, 640, 640)
    stage1: (B,  32, 320, 320)  ← stride=2
    stage2: (B,  64, 160, 160)  ← stride=2 + ResBlock×1
    stage3: (B, 128,  80,  80)  ← stride=2 + ResBlock×2
    stage4: (B, 256,  40,  40)  ← stride=2 + ResBlock×2
    stage5: (B, 512,  20,  20)  ← stride=2 + ResBlock×1
    Output: (B, 512,  20,  20)
    """

    def __init__(self) -> None:
        super().__init__()

        # Первичная обработка: 3→32, разрешение 640→320
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),   # (B,32,320,320)
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1),  # (B,32,320,320)
        )

        # 32→64, разрешение 320→160
        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1),  # (B,64,160,160)
            ResidualBlock(64),                                        # (B,64,160,160)
        )

        # 64→128, разрешение 160→80; два ResBlock для лучшего извлечения признаков
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1), # (B,128,80,80)
            ResidualBlock(128),                                       # (B,128,80,80)
            ResidualBlock(128),                                       # (B,128,80,80)
        )

        # 128→256, разрешение 80→40
        self.stage4 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1),# (B,256,40,40)
            ResidualBlock(256),                                       # (B,256,40,40)
            ResidualBlock(256),                                       # (B,256,40,40)
        )

        # 256→512, разрешение 40→20 (финальная карта признаков для детекции)
        self.stage5 = nn.Sequential(
            ConvBNReLU(256, 512, kernel_size=3, stride=2, padding=1),# (B,512,20,20)
            ResidualBlock(512),                                       # (B,512,20,20)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)  # (B,  32, 320, 320)
        x = self.stage2(x)  # (B,  64, 160, 160)
        x = self.stage3(x)  # (B, 128,  80,  80)
        x = self.stage4(x)  # (B, 256,  40,  40)
        x = self.stage5(x)  # (B, 512,  20,  20)
        return x


class DetectionHead(nn.Module):
    """
    Голова детекции: формирует предсказания для каждой ячейки сетки.

    Структура выходного вектора на ячейку (канальное измерение):
      [0]    — logit объектности (sigmoid при декодировании)
      [1..4] — tx, ty, tw, th   (sigmoid → нормализованные координаты)
      [5..]  — логиты классов   (argmax при декодировании)

    Input:  (B, 512, 20, 20)
    Output: (B, 1+4+C, 20, 20)
    """

    def __init__(self, in_ch: int = 512, num_classes: int = 3) -> None:
        super().__init__()
        out_ch = 1 + 4 + num_classes  # obj + bbox + классы

        self.head = nn.Sequential(
            ConvBNReLU(in_ch, 256, kernel_size=3, stride=1, padding=1),  # (B,256,20,20)
            ConvBNReLU(256, 128, kernel_size=1, stride=1, padding=0),    # (B,128,20,20)
            nn.Conv2d(128, out_ch, kernel_size=1, stride=1, padding=0),  # (B,out_ch,20,20)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, 1+4+C, 20, 20)


class Detector(nn.Module):
    """
    Полная модель детектора: Backbone + DetectionHead.

    Input:  (B, 3, 640, 640)        — RGB, нормализованное [0, 1]
    Output: (B, 1+4+C, 20, 20)      — сырые логиты для каждой ячейки сетки

    Принцип работы:
      Изображение делится на сетку G×G (G=20, stride=32px).
      Каждая ячейка (gi, gj) отвечает за объект, чей центр попадает в неё.
      В ходе обучения только «ответственная» ячейка получает градиент bbox/class.
    """

    GRID_SIZE: int = 20

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Backbone()
        self.head = DetectionHead(in_ch=512, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 640, 640) — батч изображений

        Returns:
            (B, 1+4+C, 20, 20) — сырые предсказания
        """
        features = self.backbone(x)  # (B, 512, 20, 20)
        out = self.head(features)    # (B, 1+4+C, 20, 20)
        return out

    def predict(
        self, x: torch.Tensor, conf_thresh: float = 0.5
    ) -> list:
        """
        Инференс: декодирует предсказания в bbox + класс.

        Args:
            x:           (B, 3, 640, 640)
            conf_thresh: минимальный порог объектности

        Returns:
            список длины B; каждый элемент:
              {'class': int, 'conf': float, 'bbox': [x_c, y_c, w, h]}
              или None, если объект не найден
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)  # (B, 1+4+C, 20, 20)

        G = self.GRID_SIZE
        results = []

        for b in range(x.shape[0]):
            pred = raw[b]  # (1+4+C, 20, 20)

            # Карта объектности после sigmoid → (20, 20)
            obj_map = torch.sigmoid(pred[0])  # (20, 20)

            max_conf = obj_map.max().item()
            if max_conf < conf_thresh:
                results.append(None)
                continue

            # Ячейка с максимальной объектностью
            flat_idx = obj_map.argmax()
            gi = (flat_idx // G).item()  # строка (ось Y)
            gj = (flat_idx %  G).item()  # столбец (ось X)

            # Декодирование bbox: sigmoid → нормализованные координаты
            tx = torch.sigmoid(pred[1, gi, gj]).item()
            ty = torch.sigmoid(pred[2, gi, gj]).item()
            tw = torch.sigmoid(pred[3, gi, gj]).item()
            th = torch.sigmoid(pred[4, gi, gj]).item()

            x_c = (gj + tx) / G
            y_c = (gi + ty) / G

            # Класс с максимальным логитом
            cls = int(pred[5:, gi, gj].argmax().item())

            results.append({
                'class': cls,
                'conf':  float(max_conf),
                'bbox':  [x_c, y_c, tw, th],
            })

        return results
