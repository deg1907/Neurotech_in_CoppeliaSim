"""
Функция потерь детектора.

Компоненты:
  1. Objectness loss — BCE: положительные ячейки (объект) vs отрицательные (фон)
  2. Bbox loss      — Smooth L1: только для положительных ячеек
  3. Class loss     — CrossEntropy: только для положительных ячеек

Веса:
  lambda_obj   — вес положительных ячеек в BCE
  lambda_noobj — вес отрицательных ячеек (<<1 для компенсации дисбаланса 1:399)
  lambda_bbox  — вес bbox loss (увеличен, т.к. точность локализации критична)
  lambda_cls   — вес классификационной loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    Суммарная loss функция детектора.

    Args:
        grid_size:    размер сетки G (по умолчанию 20)
        num_classes:  число классов
        lambda_obj:   вес obj loss для положительных ячеек
        lambda_noobj: вес obj loss для отрицательных ячеек (борьба с дисбалансом)
        lambda_bbox:  вес bbox loss
        lambda_cls:   вес class loss
    """

    def __init__(
        self,
        grid_size: int = 20,
        num_classes: int = 3,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_bbox: float = 5.0,
        lambda_cls: float = 1.0,
    ) -> None:
        super().__init__()
        self.G = grid_size
        self.num_classes = num_classes
        self.lambda_obj   = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox  = lambda_bbox
        self.lambda_cls   = lambda_cls

    def forward(
        self,
        predictions: torch.Tensor,
        targets: list,
    ) -> tuple[torch.Tensor, dict]:
        """
        Вычислить суммарную loss.

        Args:
            predictions: (B, 1+4+C, G, G) — сырые предсказания модели
            targets:     список длины B; каждый элемент dict:
                           {'class': int, 'bbox': [x_c, y_c, w, h]}
                         или None (изображение без объекта — пропускаем)

        Returns:
            total_loss: скаляр
            components: dict с ключами 'obj', 'bbox', 'cls' для логирования
        """
        B, _, G, _ = predictions.shape
        device = predictions.device

        # Маска положительных ячеек: 1 там, где есть объект
        obj_mask = torch.zeros(B, G, G, device=device)          # (B, G, G)

        # Целевые значения bbox и классов в положительных ячейках
        tx_tgt = torch.zeros(B, G, G, device=device)
        ty_tgt = torch.zeros(B, G, G, device=device)
        tw_tgt = torch.zeros(B, G, G, device=device)
        th_tgt = torch.zeros(B, G, G, device=device)
        cls_tgt = torch.zeros(B, G, G, dtype=torch.long, device=device)

        for b, tgt in enumerate(targets):
            if tgt is None:
                continue

            x_c, y_c, w, h = tgt['bbox']
            cls = int(tgt['class'])

            # Ячейка, ответственная за данный объект
            gj = int(x_c * G)  # столбец (ось X)
            gi = int(y_c * G)  # строка  (ось Y)
            gj = min(gj, G - 1)
            gi = min(gi, G - 1)

            obj_mask[b, gi, gj] = 1.0

            # Целевые смещения внутри ячейки (значения до sigmoid)
            # Поскольку sigmoid(tx_tgt) должен равняться (x_c*G - gj),
            # а loss считается уже после sigmoid предсказания,
            # используем нормализованные значения напрямую
            tx_tgt[b, gi, gj] = x_c * G - gj   # ∈ [0, 1)
            ty_tgt[b, gi, gj] = y_c * G - gi   # ∈ [0, 1)
            tw_tgt[b, gi, gj] = w               # нормализованный размер
            th_tgt[b, gi, gj] = h               # нормализованный размер
            cls_tgt[b, gi, gj] = cls

        noobj_mask = 1.0 - obj_mask  # (B, G, G)

        # ── Objectness loss ──────────────────────────────────────────────────
        # Применяем sigmoid к каналу объектности → (B, G, G)
        pred_obj = torch.sigmoid(predictions[:, 0, :, :])  # (B, G, G)

        # BCE поэлементно → отдельно взвешиваем pos/neg ячейки
        bce = F.binary_cross_entropy(pred_obj, obj_mask, reduction='none')  # (B,G,G)

        obj_loss = (
            self.lambda_obj   * (bce * obj_mask).sum() +
            self.lambda_noobj * (bce * noobj_mask).sum()
        ) / B

        # ── Bbox loss (только положительные ячейки) ─────────────────────────
        # Предсказания bbox после sigmoid → (B, 4, G, G)
        pred_bbox = torch.sigmoid(predictions[:, 1:5, :, :])  # (B, 4, G, G)

        # Собираем цели в один тензор (B, 4, G, G)
        bbox_tgt = torch.stack([tx_tgt, ty_tgt, tw_tgt, th_tgt], dim=1)

        # Smooth L1 поэлементно, маскируем и усредняем по числу объектов
        bbox_loss_map = F.smooth_l1_loss(pred_bbox, bbox_tgt, reduction='none')
        # Суммируем по 4 координатам → (B, G, G)
        bbox_loss_map = bbox_loss_map.sum(dim=1)

        n_obj = obj_mask.sum().clamp(min=1.0)
        bbox_loss = self.lambda_bbox * (bbox_loss_map * obj_mask).sum() / n_obj

        # ── Class loss (только положительные ячейки) ────────────────────────
        # predictions[:, 5:, :, :] — логиты классов (B, C, G, G)
        pred_cls = predictions[:, 5:, :, :]  # (B, C, G, G)

        # Собираем только положительные ячейки
        pos_indices = obj_mask.bool()  # (B, G, G) bool маска

        if pos_indices.any():
            # pred_cls: (B, C, G, G) → транспонируем для удобства
            # Выбираем логиты в положительных ячейках
            pred_cls_flat = pred_cls.permute(0, 2, 3, 1)  # (B, G, G, C)
            pred_pos = pred_cls_flat[pos_indices]          # (N_obj, C)
            cls_pos  = cls_tgt[pos_indices]                # (N_obj,)
            cls_loss = self.lambda_cls * F.cross_entropy(pred_pos, cls_pos)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = obj_loss + bbox_loss + cls_loss

        return total_loss, {
            'obj':  obj_loss.item(),
            'bbox': bbox_loss.item(),
            'cls':  cls_loss.item(),
        }
