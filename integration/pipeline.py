"""
Полный inference pipeline: кадр → детекция → позиция → захват.

Последовательность:
  1. Захват кадра 640×640 с Vision Sensor
  2. Детектор (CNN) → bbox (x_c, y_c, w, h) + class
  3. Pixel → World координаты через обратную проекцию (calibration.py)
  4. Yaw — заглушка 0.0° (TODO: подключить yaw-регрессор)
  5. Возврат (class_id, x_world, y_world, yaw_deg)

Атрибут last_bbox_px хранит bbox последнего предсказания в пикселях —
используется в main_loop.py для передачи в DepthEstimator.

Использование в main_loop.py:
  pipeline = DetectionPipeline(weights_path, sim, vision_handle)
  class_id, x_w, y_w, yaw = pipeline.run(frame)
  bbox_px = pipeline.last_bbox_px  # (x1, y1, x2, y2)
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Пути к модулям детектора и калибровки
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'detector'))
sys.path.insert(0, str(Path(__file__).parent))

from architecture import Detector
from calibration import pixel_to_world


class DetectionPipeline:
    """
    Объединяет детектор и калибровку в один вызов.

    Args:
        weights_path:  путь к .pt файлу весов детектора
        sim:           объект Remote API CoppeliaSim
        vision_handle: handle Vision Sensor
        num_classes:   число классов
        conf_thresh:   порог объектности детектора
        device:        'cpu' или 'cuda'
    """

    def __init__(
        self,
        weights_path: str,
        sim,
        vision_handle: int,
        num_classes: int = 3,
        conf_thresh: float = 0.5,
        device: str = 'cpu',
    ) -> None:
        self.sim           = sim
        self.vision_handle = vision_handle
        self.conf_thresh   = conf_thresh
        self.device        = torch.device(device)

        # Загрузка детектора
        self.model = Detector(num_classes=num_classes).to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        print(f'[Pipeline] Детектор загружен: {weights_path}')

        # bbox последнего предсказания в пикселях (x1, y1, x2, y2)
        # Используется в main_loop.py для DepthEstimator
        self.last_bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)

    def run(self, frame: np.ndarray) -> tuple[int, float, float, float]:
        """
        Запустить pipeline на одном кадре.

        Args:
            frame: numpy array (640, 640, 3), RGB, uint8

        Returns:
            (class_id, x_world, y_world, yaw_deg)
            class_id = -1 если деталь не найдена

        Побочный эффект:
            self.last_bbox_px обновляется bbox'ом в пикселях (x1, y1, x2, y2)
        """
        img_h, img_w = frame.shape[:2]

        # ── Препроцессинг ────────────────────────────────────────────────────
        img = frame.astype(np.float32) / 255.0          # нормализация [0,1]
        tensor = torch.from_numpy(img).permute(2, 0, 1) # (3,H,W)
        tensor = tensor.unsqueeze(0).to(self.device)    # (1,3,H,W)

        # ── Детекция ─────────────────────────────────────────────────────────
        results = self.model.predict(tensor, conf_thresh=self.conf_thresh)
        result  = results[0]

        if result is None:
            self.last_bbox_px = (0, 0, 0, 0)
            return -1, 0.0, 0.0, 0.0

        class_id = result['class']
        x_c, y_c, w, h = result['bbox']

        # ── Bbox в пикселях (для DepthEstimator) ─────────────────────────────
        x1 = max(0, int((x_c - w / 2) * img_w))
        y1 = max(0, int((y_c - h / 2) * img_h))
        x2 = min(img_w, int((x_c + w / 2) * img_w))
        y2 = min(img_h, int((y_c + h / 2) * img_h))
        self.last_bbox_px = (x1, y1, x2, y2)

        # ── Pixel → World ─────────────────────────────────────────────────────
        u = x_c * float(img_w)
        v = y_c * float(img_h)
        x_world, y_world = pixel_to_world(u, v, self.sim, self.vision_handle)

        # ── Yaw (заглушка) ───────────────────────────────────────────────────
        # TODO: заменить на вызов yaw-регрессора
        yaw_deg = 0.0

        return class_id, x_world, y_world, yaw_deg
