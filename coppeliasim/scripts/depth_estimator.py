"""
Оценка высоты детали над конвейером из карты глубины.

Алгоритм:
  1. Вырезать область depth-карты по bbox детали (из детектора).
  2. Убрать фоновые пиксели (глубина ≈ baseline пустой ленты).
  3. Взять 5-й перцентиль оставшихся значений = самая близкая к камере точка
     = верхняя поверхность детали.
  4. Высота = baseline_depth - top_depth.

Calibrate:
  При старте системы, до подачи первой детали, вызвать calibrate_belt()
  с пустым конвейером — метод сохранит baseline глубины.

Использование:
  depth_est = DepthEstimator(sim, depth_sensor.handle)
  depth_est.calibrate_belt(depth_sensor.capture_depth())

  # В цикле сортировки:
  depth_frame = depth_sensor.capture_depth()
  height_m = depth_est.estimate_height(depth_frame, bbox_px=(x1, y1, x2, y2))
"""

import numpy as np


class DepthEstimator:
    """
    Оценка высоты верхней поверхности детали над конвейерной лентой.

    Использует depth Vision Sensor (аналог Intel RealSense D435i).
    Не требует знания класса или ориентации детали — результат универсален.
    """

    # Запасной baseline если calibrate_belt не был вызван (м).
    # НАСТРОИТЬ: измерить вручную или через calibrate_belt().
    DEFAULT_BELT_DEPTH: float = 1.2

    # Процент "самых близких" пикселей для оценки верхней грани.
    TOP_PERCENTILE: float = 5.0

    # Пороговый коэффициент для отсечения фоновых пикселей.
    # 0.995 = ищем пиксели ближе чем 99.5% от дальности ленты.
    # Увеличено с 0.98: обнаруживает тонкие детали и устраняет проблему
    # несовпадения пикселей RGB/depth сенсоров.
    BACKGROUND_THRESHOLD: float = 0.995

    # Минимальный размер bbox для надёжной оценки (пикселей по одной стороне).
    MIN_BBOX_SIZE: int = 10

    def __init__(self) -> None:
        """Инициализация. Вызвать calibrate_belt() перед использованием."""
        self.belt_depth_m: float = self.DEFAULT_BELT_DEPTH
        self._calibrated: bool = False

    def calibrate_belt(self, depth_frame: np.ndarray) -> float:
        """
        Замерить baseline глубины пустого конвейера.

        Вызывать один раз при старте, пока на конвейере нет деталей.
        Берёт median центрального региона (60% ширины и высоты) —
        чтобы не учитывать края рамки сцены.

        Args:
            depth_frame: карта глубины (H, W) float32 в метрах

        Returns:
            замеренная базовая глубина (м) — поверхность ленты
        """
        h, w = depth_frame.shape
        # Центральный регион: 20%..80% по каждой оси
        y1 = int(h * 0.20)
        y2 = int(h * 0.80)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        center_region = depth_frame[y1:y2, x1:x2]
        self.belt_depth_m = float(np.median(center_region))
        self._calibrated = True
        return self.belt_depth_m

    def estimate_height_global(self, depth_frame: np.ndarray) -> float:
        """
        Оценить высоту детали сканируя весь кадр — без bbox.

        Используется когда RGB и depth сенсоры не совпадают попиксельно
        (разные позиции или FOV в сцене). На конвейере одна деталь,
        поэтому самая близкая к камере точка = верхняя грань детали.

        Args:
            depth_frame: карта глубины (H, W) float32 в метрах

        Returns:
            высота верхней грани над лентой в метрах (>= 0)
        """
        background_thresh = self.belt_depth_m * self.BACKGROUND_THRESHOLD
        part_mask = depth_frame < background_thresh

        if part_mask.sum() < 5:
            return 0.0

        top_depth = float(np.percentile(depth_frame[part_mask], self.TOP_PERCENTILE))
        return max(0.0, self.belt_depth_m - top_depth)

    def estimate_height(
        self,
        depth_frame: np.ndarray,
        bbox_px: tuple[int, int, int, int],
    ) -> float:
        """
        Оценить высоту верхней поверхности детали над лентой (м).

        Args:
            depth_frame: карта глубины (H, W) float32 в метрах
            bbox_px:     (x1, y1, x2, y2) — bbox детали в пикселях,
                         из вывода детектора (postprocess → bbox_px)

        Returns:
            высота верхней грани над лентой в метрах (>= 0).
            0.0 если bbox слишком мал или нет валидных пикселей.
        """
        x1, y1, x2, y2 = bbox_px

        # Зажать bbox по границам кадра
        h, w = depth_frame.shape
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        # Проверка: bbox достаточно большой
        if (x2 - x1) < self.MIN_BBOX_SIZE or (y2 - y1) < self.MIN_BBOX_SIZE:
            return 0.0

        # Вырезать регион
        crop = depth_frame[y1:y2, x1:x2]  # (crop_h, crop_w) float32

        # Убрать фоновые пиксели (глубина ≈ ленте или дальше)
        background_thresh = self.belt_depth_m * self.BACKGROUND_THRESHOLD
        part_mask = crop < background_thresh

        if part_mask.sum() < 5:
            # Слишком мало пикселей детали в регионе
            return 0.0

        # TOP_PERCENTILE% ближайших к камере пикселей = верхняя грань
        part_depths = crop[part_mask]
        top_depth = float(np.percentile(part_depths, self.TOP_PERCENTILE))

        # Высота = baseline (лента) - глубина верхней грани
        height = self.belt_depth_m - top_depth
        return max(0.0, height)

    def estimate_height_stats(
        self,
        depth_frame: np.ndarray,
        bbox_px: tuple[int, int, int, int],
    ) -> dict:
        """
        Расширенная статистика глубины в bbox (для отладки и калибровки).

        Returns:
            dict с ключами: 'height_m', 'top_depth', 'belt_depth',
                            'n_part_pixels', 'calibrated'
        """
        height = self.estimate_height(depth_frame, bbox_px)
        x1, y1, x2, y2 = bbox_px
        h, w = depth_frame.shape
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        crop = depth_frame[y1:y2, x1:x2]
        bg_thresh = self.belt_depth_m * self.BACKGROUND_THRESHOLD
        part_mask = crop < bg_thresh
        n_part = int(part_mask.sum())
        top_depth = float(np.percentile(crop[part_mask], self.TOP_PERCENTILE)) if n_part >= 5 else 0.0

        return {
            'height_m':      height,
            'top_depth':     top_depth,
            'belt_depth':    self.belt_depth_m,
            'n_part_pixels': n_part,
            'calibrated':    self._calibrated,
        }
