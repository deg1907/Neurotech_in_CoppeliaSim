"""
Калибровка камера → робот: pixel → world координаты.

Метод: обратная проекция пикселя через pinhole-модель камеры.
Поскольку деталь лежит на ленте конвейера (z = BELT_Z = const),
луч из камеры пересекается с горизонтальной плоскостью z = BELT_Z.

Математика (обратная к annotator.py):
  1. Получить матрицу позы камеры из CoppeliaSim (getObjectMatrix)
  2. Перевести (u, v) в нормализованные координаты камеры (xc, yc)
  3. Построить луч в мировой СК: R^T @ [xc, yc, 1]
  4. Найти t: camera_z + t * ray_z = BELT_Z
  5. x_world = camera_x + t * ray_x
     y_world = camera_y + t * ray_y
"""

import math
import numpy as np


# Высота поверхности ленты конвейера в мировой СК (м)
BELT_Z: float = 0.131


def get_camera_intrinsics(sim, vision_handle: int, img_w: int = 640, img_h: int = 640
                          ) -> tuple[float, float, float, float]:
    """
    Получить параметры pinhole-модели Vision Sensor.

    Returns:
        (fx, fy, cx, cy) в пикселях
    """
    try:
        fov_rad = sim.getObjectFloatParam(
            vision_handle, sim.visionfloatparam_perspective_angle
        )
    except Exception:
        fov_rad = math.radians(30.0)

    fx = (img_w / 2.0) / math.tan(fov_rad / 2.0)
    fy = (img_h / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    return fx, fy, cx, cy


def pixel_to_world(
    u: float,
    v: float,
    sim,
    vision_handle: int,
    belt_z: float = BELT_Z,
    img_w: int = 640,
    img_h: int = 640,
) -> tuple[float, float]:
    """
    Перевести пиксельные координаты центра bbox в мировые (x, y).

    Args:
        u, v:          координаты пикселя (центр bbox)
        sim:           объект Remote API
        vision_handle: handle Vision Sensor
        belt_z:        высота поверхности ленты (м)
        img_w, img_h:  размер изображения

    Returns:
        (x_world, y_world) в метрах
    """
    fx, fy, cx, cy = get_camera_intrinsics(sim, vision_handle, img_w, img_h)

    # Матрица позы камеры в мировой СК (3×4)
    m = sim.getObjectMatrix(vision_handle, -1)
    M = np.array(m).reshape(3, 4)
    R = M[:, :3]   # матрица вращения (camera → world)
    cam_pos = M[:, 3]  # позиция камеры в мировой СК

    # Нормализованные координаты в СК камеры
    # Примечание: ось X камеры инвертирована (см. annotator.py)
    xc = -(u - cx) / fx   # минус: компенсируем инверсию из annotator.py
    yc = -(v - cy) / fy   # минус: ось Y камеры направлена вверх

    # Направление луча в мировой СК: R @ [xc, yc, 1]
    ray_cam   = np.array([xc, yc, 1.0])
    ray_world = R @ ray_cam  # (3,)

    # Пересечение с плоскостью z = belt_z
    # cam_pos[2] + t * ray_world[2] = belt_z
    if abs(ray_world[2]) < 1e-6:
        # Луч параллелен плоскости — не должно происходить при правильной установке камеры
        return float(cam_pos[0]), float(cam_pos[1])

    t = (belt_z - cam_pos[2]) / ray_world[2]

    x_world = cam_pos[0] + t * ray_world[0]
    y_world = cam_pos[1] + t * ray_world[1]

    return float(x_world), float(y_world)
