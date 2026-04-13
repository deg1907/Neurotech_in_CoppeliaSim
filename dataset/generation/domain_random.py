"""
Domain Randomization для ablation study.

Три уровня рандомизации (выбираются при генерации датасета):
  DR_NONE    — без рандомизации (baseline)
  DR_LIGHT   — только освещение (partial DR)
  DR_FULL    — освещение + цвет деталей + шум камеры (full DR)

Освещение рандомизируется на уровне сцены CoppeliaSim (через API).
Шум камеры и цвет деталей — постобработка numpy-кадра / свойства объекта.
"""

from __future__ import annotations
from enum import Enum, auto
import numpy as np


# ------------------------------------------------------------------ #
#  Режимы domain randomization                                        #
# ------------------------------------------------------------------ #

class DRMode(Enum):
    NONE  = auto()   # baseline — никакой рандомизации
    LIGHT = auto()   # partial DR — только освещение
    FULL  = auto()   # full DR — освещение + цвет + шум


# ------------------------------------------------------------------ #
#  Рандомизация освещения (in-simulation)                             #
# ------------------------------------------------------------------ #

class LightRandomizer:
    """
    Рандомизирует параметры источников света в сцене CoppeliaSim.
    Находит все объекты-лампы в сцене и изменяет их яркость и цвет.
    """

    # Диапазоны рандомизации
    INTENSITY_RANGE: tuple[float, float] = (0.4, 1.5)   # множитель яркости
    COLOR_JITTER:    float               = 0.15          # ±отклонение RGB [0,1]

    def __init__(self, sim) -> None:
        self.sim = sim
        self._light_handles: list[int] = []
        self._original_colors: dict[int, list[float]] = {}
        self._find_lights()

    def _find_lights(self) -> None:
        """Найти все источники света в сцене."""
        index = 0
        while True:
            h = self.sim.getObjects(index, self.sim.object_light_type)
            if h == -1:
                break
            self._light_handles.append(h)
            # Сохранить исходный цвет (diffuse)
            color = self.sim.getObjectColor(
                h, 0, self.sim.colorcomponent_ambient_diffuse
            )
            self._original_colors[h] = color
            index += 1

    def randomize(self) -> None:
        """Применить случайные параметры освещения."""
        rng = np.random.default_rng()
        for h in self._light_handles:
            base = self._original_colors[h]
            intensity = rng.uniform(*self.INTENSITY_RANGE)
            jitter = rng.uniform(-self.COLOR_JITTER, self.COLOR_JITTER, 3)
            new_color = np.clip(np.array(base) * intensity + jitter, 0.0, 1.0)
            self.sim.setObjectColor(
                h, 0, self.sim.colorcomponent_ambient_diffuse,
                new_color.tolist()
            )

    def reset(self) -> None:
        """Вернуть исходные параметры освещения."""
        for h in self._light_handles:
            self.sim.setObjectColor(
                h, 0, self.sim.colorcomponent_ambient_diffuse,
                self._original_colors[h]
            )


# ------------------------------------------------------------------ #
#  Рандомизация цвета деталей (in-simulation)                         #
# ------------------------------------------------------------------ #

# Базовые цвета деталей по классу (RGB)
_BASE_COLORS: dict[int, list[float]] = {
    0: [0.70, 0.70, 0.75],  # gaika  — сталь/серебро
    1: [0.65, 0.65, 0.70],  # vilka  — тёмная сталь
    2: [0.80, 0.82, 0.85],  # vtulka — алюминий
}

# Диапазон отклонения цвета
_COLOR_JITTER = 0.20


def randomize_part_color(sim, part_handle: int, class_id: int) -> None:
    """
    Применить случайный цвет к детали (в пределах реалистичного диапазона).

    Args:
        sim:         Remote API
        part_handle: handle детали
        class_id:    0=gaika, 1=vilka, 2=vtulka
    """
    base = np.array(_BASE_COLORS.get(class_id, [0.6, 0.6, 0.6]))
    jitter = np.random.uniform(-_COLOR_JITTER, _COLOR_JITTER, 3)
    color = np.clip(base + jitter, 0.0, 1.0).tolist()
    sim.setShapeColor(
        part_handle, None, sim.colorcomponent_ambient_diffuse, color
    )


def reset_part_color(sim, part_handle: int, class_id: int) -> None:
    """Вернуть базовый цвет детали."""
    color = _BASE_COLORS.get(class_id, [0.6, 0.6, 0.6])
    sim.setShapeColor(
        part_handle, None, sim.colorcomponent_ambient_diffuse, color
    )


# ------------------------------------------------------------------ #
#  Рандомизация изображения (post-processing numpy)                   #
# ------------------------------------------------------------------ #

def add_gaussian_noise(
    frame: np.ndarray,
    sigma_range: tuple[float, float] = (2.0, 15.0),
) -> np.ndarray:
    """
    Добавить гауссов шум к изображению.

    Args:
        frame:       numpy array (H, W, 3), uint8
        sigma_range: диапазон стандартного отклонения шума (пиксели)

    Returns:
        зашумлённое изображение (H, W, 3), uint8
    """
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, frame.shape)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def random_brightness(
    frame: np.ndarray,
    factor_range: tuple[float, float] = (0.6, 1.4),
) -> np.ndarray:
    """
    Случайная яркость (на уровне изображения, дополняет рандомизацию освещения).

    Args:
        frame:        numpy array (H, W, 3), uint8
        factor_range: диапазон множителя яркости

    Returns:
        изображение с изменённой яркостью, uint8
    """
    factor = np.random.uniform(*factor_range)
    adjusted = np.clip(frame.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)


# ------------------------------------------------------------------ #
#  Единая точка входа                                                  #
# ------------------------------------------------------------------ #

def apply_image_dr(frame: np.ndarray, mode: DRMode) -> np.ndarray:
    """
    Применить image-level domain randomization в зависимости от режима.

    Args:
        frame: numpy array (H, W, 3), uint8, RGB
        mode:  режим DR

    Returns:
        обработанный кадр (H, W, 3), uint8
    """
    if mode == DRMode.NONE:
        return frame

    if mode in (DRMode.LIGHT, DRMode.FULL):
        # Лёгкая яркость-аугментация на уровне изображения
        # (основная рандомизация освещения — через LightRandomizer)
        frame = random_brightness(frame, factor_range=(0.8, 1.2))

    if mode == DRMode.FULL:
        frame = add_gaussian_noise(frame, sigma_range=(2.0, 12.0))

    return frame
