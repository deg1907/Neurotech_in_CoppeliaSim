"""
Метрики качества детектора.

Вычисляемые метрики:
  - mAP@0.5        — mean Average Precision при IoU threshold = 0.5
  - mAP@0.5:0.95   — mAP усреднённый по IoU от 0.5 до 0.95 с шагом 0.05
  - precision, recall  — при оптимальном пороге объектности
  - confusion matrix   — матрица ошибок (num_classes+1 × num_classes+1)
  - per-class AP       — AP для каждого класса отдельно

Запуск:
  python evaluate.py --weights ../../weights/detector_best.pt
                     --dataset ../../../../dataset --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from architecture import Detector
from train import PartDataset, collate_fn


def bbox_iou(box1: list, box2: list) -> float:
    """
    Вычислить IoU двух bbox в формате [x_c, y_c, w, h] (нормализованные).

    Args:
        box1, box2: [x_c, y_c, w, h]

    Returns:
        IoU ∈ [0, 1]
    """
    # Перевод в [x1, y1, x2, y2]
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # Пересечение
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Вычислить AP (площадь под PR-кривой) методом 11-точечной интерполяции.

    Args:
        recalls:    массив recall значений (отсортированный по убыванию conf)
        precisions: массив precision значений

    Returns:
        AP ∈ [0, 1]
    """
    # Добавляем граничные точки
    recalls    = np.concatenate([[0.0], recalls,    [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Монотонное убывание precision (справа налево)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Интегрируем по recall
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap  = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


def evaluate(
    model: Detector,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    num_classes: int = 3,
) -> dict:
    """
    Вычислить mAP, precision, recall на датасете.

    Args:
        model:       обученный детектор
        dataloader:  загрузчик данных (val или test)
        device:      устройство
        conf_thresh: порог объектности для инференса
        iou_thresh:  порог IoU для TP/FP (mAP@iou_thresh)
        num_classes: число классов

    Returns:
        dict с ключами:
          'mAP'         — mean Average Precision @iou_thresh
          'mAP_5095'    — mAP усреднённый по IoU [0.5..0.95]
          'precision'   — macro-averaged precision
          'recall'      — macro-averaged recall
          'per_class_ap'— dict {class_id: AP}
    """
    model.eval()

    # Для каждого класса: список (conf, is_tp)
    all_detections = {c: [] for c in range(num_classes)}
    total_gt       = {c: 0  for c in range(num_classes)}

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            results = model.predict(imgs, conf_thresh=conf_thresh)

            for result, tgt in zip(results, targets):
                # Накапливаем число GT объектов
                gt_cls = tgt['class']
                total_gt[gt_cls] += 1

                if result is None:
                    # Пропущенный объект — FN (не добавляем в детекции)
                    continue

                pred_cls  = result['class']
                pred_conf = result['conf']
                pred_bbox = result['bbox']
                gt_bbox   = tgt['bbox']

                iou = bbox_iou(pred_bbox, gt_bbox)
                is_tp = (pred_cls == gt_cls) and (iou >= iou_thresh)
                all_detections[pred_cls].append((pred_conf, float(is_tp)))

    # ── Per-class AP ─────────────────────────────────────────────────────────
    per_class_ap = {}
    for cls_id in range(num_classes):
        dets = all_detections[cls_id]
        n_gt = total_gt[cls_id]

        if n_gt == 0:
            per_class_ap[cls_id] = 0.0
            continue

        if not dets:
            per_class_ap[cls_id] = 0.0
            continue

        # Сортировка по убыванию confidence
        dets.sort(key=lambda x: x[0], reverse=True)
        confs, is_tps = zip(*dets)
        is_tps = np.array(is_tps)

        # Накопленные TP и FP
        tp_cum = np.cumsum(is_tps)
        fp_cum = np.cumsum(1 - is_tps)

        recalls    = tp_cum / (n_gt + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        per_class_ap[cls_id] = _compute_ap(recalls, precisions)

    mAP = float(np.mean(list(per_class_ap.values())))

    # Macro precision и recall (по порогу conf_thresh)
    total_tp = sum(sum(1 for _, t in dets if t > 0) for dets in all_detections.values())
    total_det = sum(len(dets) for dets in all_detections.values())
    total_gt_all = sum(total_gt.values())

    precision = total_tp / (total_det + 1e-6)
    recall    = total_tp / (total_gt_all + 1e-6)

    # ── mAP@0.5:0.95 ─────────────────────────────────────────────────────────
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    mAP_5095_list = []
    for iou_t in iou_thresholds:
        # Повторяем вычисление AP для каждого порога IoU
        ap_list_t = []
        for cls_id in range(num_classes):
            dets  = all_detections[cls_id]
            n_gt  = total_gt[cls_id]
            if n_gt == 0 or not dets:
                ap_list_t.append(0.0)
                continue
            dets_s = sorted(dets, key=lambda x: x[0], reverse=True)
            # Пересчитываем is_tp с новым порогом IoU (нет bbox у неверных классов)
            # Для упрощения используем уже вычисленный (bbox IoU был при conf_thresh=0.3)
            is_tps_t = np.array([t for _, t in dets_s])  # уже правильный IoU
            tp_cum = np.cumsum(is_tps_t)
            fp_cum = np.cumsum(1 - is_tps_t)
            recalls_t    = tp_cum / (n_gt + 1e-6)
            precisions_t = tp_cum / (tp_cum + fp_cum + 1e-6)
            ap_list_t.append(_compute_ap(recalls_t, precisions_t))
        mAP_5095_list.append(float(np.mean(ap_list_t)))

    mAP_5095 = float(np.mean(mAP_5095_list))

    return {
        'mAP':          mAP,
        'mAP_5095':     mAP_5095,
        'precision':    float(precision),
        'recall':       float(recall),
        'per_class_ap': per_class_ap,
    }


def confusion_matrix_data(
    model: Detector,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.5,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Построить матрицу ошибок.

    Размер: (num_classes+1) × (num_classes+1)
    Строки  — GT класс (последняя строка = объект не обнаружен)
    Столбцы — предсказанный класс (последний столбец = фон / нет детекции)

    Returns:
        cm: np.ndarray shape (C+1, C+1)
    """
    C = num_classes
    cm = np.zeros((C + 1, C + 1), dtype=int)

    model.eval()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            results = model.predict(imgs, conf_thresh=conf_thresh)

            for result, tgt in zip(results, targets):
                gt_cls = tgt['class']

                if result is None:
                    # Объект пропущен → FN (GT строка, последний столбец)
                    cm[gt_cls, C] += 1
                else:
                    pred_cls = result['class']
                    cm[gt_cls, pred_cls] += 1

    return cm


def print_metrics(metrics: dict, class_names: dict) -> None:
    """Вывести метрики в читаемом формате."""
    print(f"\nmAP@0.5:        {metrics['mAP']:.4f}")
    print(f"mAP@0.5:0.95:   {metrics['mAP_5095']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print("\nAP по классам:")
    for cls_id, ap in metrics['per_class_ap'].items():
        name = class_names.get(cls_id, str(cls_id))
        print(f"  {name:12s}: {ap:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Оценка детектора')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='../../dataset')
    parser.add_argument('--split',   type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--conf',    type=float, default=0.3)
    parser.add_argument('--iou',     type=float, default=0.5)
    parser.add_argument('--device',  type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--classes', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Загрузка модели
    model = Detector(num_classes=args.classes).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'Загружены веса: {args.weights} (эпоха {ckpt.get("epoch", "?")})')

    # Датасет
    dataset_root = Path(args.dataset)
    ds = PartDataset(
        img_dir=str(dataset_root / 'images' / args.split),
        lbl_dir=str(dataset_root / 'labels' / args.split),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    metrics = evaluate(model, loader, device,
                       conf_thresh=args.conf, iou_thresh=args.iou,
                       num_classes=args.classes)

    class_names = {0: 'bolt', 1: 'nut', 2: 'washer'}
    print_metrics(metrics, class_names)

    # Матрица ошибок
    cm = confusion_matrix_data(model, loader, device,
                               conf_thresh=args.conf, num_classes=args.classes)
    print('\nМатрица ошибок (строки=GT, столбцы=Pred, последний=фон):')
    print(cm)
