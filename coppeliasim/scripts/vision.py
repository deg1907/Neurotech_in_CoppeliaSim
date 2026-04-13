"""
Захват кадров и карт глубины с Vision Sensor CoppeliaSim.

VisionSensor — захват RGB-изображений (640×640, uint8).
VisionSensorDepth — захват карты глубины (640×640, float32, метры).

Vision Sensor настроен с "Explicit handling" = ON, поэтому рендер
происходит только по явному вызову sim.handleVisionSensor().

CoppeliaSim возвращает изображение перевёрнутым по оси Y — исправляется
через np.flipud().

Реальный аналог VisionSensorDepth: Intel RealSense D435i (structured-light,
range ≤10м, погрешность ~0.1мм при 1м).
"""

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class VisionSensor:
    """Обёртка над Vision Sensor CoppeliaSim."""

    # Путь к Vision Sensor в сцене
    SENSOR_PATH: str = '/vision_sensor_main'

    # Ожидаемое разрешение (для валидации)
    EXPECTED_W: int = 640
    EXPECTED_H: int = 640

    def __init__(self, sim) -> None:
        """
        Args:
            sim: объект Remote API
        """
        self.sim = sim
        self.handle: int = sim.getObject(self.SENSOR_PATH)

    def capture(self) -> np.ndarray:
        """
        Захватить один кадр с Vision Sensor.

        Последовательность:
          1. sim.handleVisionSensor — триггер рендера
          2. sim.getVisionSensorImg — получить пиксели (bytes, RGB)
          3. Конвертация в numpy array (H, W, 3)
          4. Переворот по Y (CoppeliaSim хранит снизу вверх)

        Returns:
            numpy array формы (640, 640, 3), dtype=uint8, RGB
        """
        # Триггер рендера
        self.sim.handleVisionSensor(self.handle)

        # Получить сырые байты изображения
        img_bytes, [width, height] = self.sim.getVisionSensorImg(self.handle)

        # Конвертация bytes → numpy
        # CoppeliaSim возвращает RGB, row-major, width*height*3 байт
        frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 3)

        # CoppeliaSim хранит первую строку снизу → переворачиваем
        frame = np.flipud(frame).copy()

        return frame  # (H, W, 3), RGB, uint8

    def capture_bgr(self) -> np.ndarray:
        """
        Захватить кадр в формате BGR (для OpenCV).

        Returns:
            numpy array (640, 640, 3), dtype=uint8, BGR
        """
        rgb = self.capture()
        return rgb[:, :, ::-1]  # RGB → BGR

    def capture_normalized(self) -> np.ndarray:
        """
        Захватить кадр, нормализованный в [0, 1] (для подачи в нейросеть).

        Returns:
            numpy array (640, 640, 3), dtype=float32, RGB, значения [0..1]
        """
        frame = self.capture().astype(np.float32) / 255.0
        return frame

    def capture_tensor(self) -> np.ndarray:
        """
        Захватить кадр в формате (C, H, W) float32 для PyTorch.

        Returns:
            numpy array (3, 640, 640), dtype=float32, значения [0..1]
        """
        frame = self.capture_normalized()      # (H, W, 3)
        return np.transpose(frame, (2, 0, 1))  # (3, H, W)


class VisionSensorDepth:
    """
    Захват карты глубины с depth Vision Sensor CoppeliaSim.

    Depth Vision Sensor настраивается в режиме "depth buffer" —
    возвращает нормированные значения [0, 1], которые пересчитываются
    в метры через near/far clipping planes.

    Реальный аналог: Intel RealSense D435i — structured-light depth camera,
    используемая в промышленных системах захвата деталей манипулятором.
    """

    # Путь к depth Vision Sensor в сцене (добавить вручную в CoppeliaSim)
    SENSOR_PATH: str = '/vision_sensor_depth'

    def __init__(self, sim) -> None:
        """
        Args:
            sim: объект Remote API
        """
        self.sim = sim
        self.handle: int = sim.getObject(self.SENSOR_PATH)

        # Читаем near/far clipping planes из свойств сенсора
        # Используются для пересчёта нормированной глубины в метры.
        self._near: float = sim.getObjectFloatParam(
            self.handle,
            sim.visionfloatparam_near_clipping,
        )
        self._far: float = sim.getObjectFloatParam(
            self.handle,
            sim.visionfloatparam_far_clipping,
        )

    def capture_depth(self) -> np.ndarray:
        """
        Захватить карту глубины.

        Последовательность:
          1. sim.handleVisionSensor — триггер рендера depth буфера
          2. sim.getVisionSensorDepth — получить нормированные float32 значения [0, 1]
          3. Пересчёт в метры: depth_m = near + (far - near) * depth_norm
          4. Reshape (W, H) → (H, W) + flipud (как в RGB-сенсоре)

        Returns:
            numpy array (640, 640), dtype=float32
            Значения в метрах: расстояние от сенсора до объекта.
            Фоновые пиксели (нет объекта) = far (максимальное расстояние).
        """
        # Триггер рендера
        self.sim.handleVisionSensor(self.handle)

        # getVisionSensorDepth возвращает (data, [width, height])
        # data может быть bytes (packed float32) или list[float] — зависит от версии API
        depth_raw, [width, height] = self.sim.getVisionSensorDepth(self.handle)

        # Конвертация в numpy:
        # bytes → np.frombuffer (интерпретирует байты как float32)
        # list  → np.array (стандартный путь)
        if isinstance(depth_raw, (bytes, bytearray)):
            depth_norm = np.frombuffer(depth_raw, dtype=np.float32).reshape(height, width)
        else:
            depth_norm = np.array(depth_raw, dtype=np.float32).reshape(height, width)
        depth_m = self._near + (self._far - self._near) * depth_norm

        # CoppeliaSim хранит первую строку снизу → переворачиваем
        depth_m = np.flipud(depth_m).copy()

        return depth_m  # (H, W), float32, метры
