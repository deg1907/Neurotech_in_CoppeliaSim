"""
Планировщик захвата: подбирает тактику захвата по высоте детали и её yaw.

Пресеты хранятся в grasp_presets.yaml — при смене деталей изменяется
только YAML, код этого модуля остаётся нетронутым.

Алгоритм resolve():
  1. По class_id → имя класса → список пресетов из YAML.
  2. Найти первый пресет с height_min ≤ height_m ≤ height_max.
  3. Вычислить ориентацию gripper (euler) из tilt_deg и yaw_deg детали.
  4. Применить contact_offset с учётом yaw: смещение [dx, dy] поворачивается
     на yaw_rad вокруг Z, чтобы всегда указывало относительно тела детали.
  5. Вернуть GraspParams(pos, euler).

Euler соглашение: CoppeliaSim, ZYX Эйлер (alpha, beta, gamma).
  gamma = yaw_deg (поворот вокруг Z мира = ориентация захвата по детали)
  beta  = -tilt_deg (наклон по оси Y → наклон gripper вперёд-назад)
  alpha = 0 (нет крена)

Tilt направление: tilt выполняется в плоскости, перпендикулярной yaw-оси,
то есть gripper наклоняется "в сторону задней стенки" относительно детали.
"""

import math
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GraspParams:
    """Параметры захвата детали: позиция и ориентация TCP gripper."""

    pos: list[float]    # [x, y, z] точка подхода gripper (мировая СК, метры)
    euler: list[float]  # [alpha, beta, gamma] ориентация gripper (радианы, ZYX)
    preset_name: str = ''  # название пресета (для логирования)


class GraspPlanner:
    """
    Универсальный планировщик захвата деталей вакуумным захватом.

    Читает пресеты из grasp_presets.yaml и по высоте детали (из depth-сенсора)
    и её yaw (из yaw-регрессора) вычисляет позу gripper.

    При замене деталей — обновляется только grasp_presets.yaml.
    """

    # Маппинг class_id → имя класса (должен совпадать с dataset.yaml)
    CLASS_NAMES: dict[int, str] = {0: 'gaika', 1: 'vilka', 2: 'vtulka'}

    # Высота TCP gripper над точкой контакта (standoff distance, метры).
    # Gripper начинает снижение с этого отступа над деталью.
    GRIPPER_STANDOFF: float = 0.005

    def __init__(self, presets_path: Optional[str] = None) -> None:
        """
        Args:
            presets_path: путь к grasp_presets.yaml.
                          По умолчанию — граsp_presets.yaml рядом с этим файлом.
        """
        if presets_path is None:
            presets_path = str(Path(__file__).parent / 'grasp_presets.yaml')

        with open(presets_path, 'r', encoding='utf-8') as f:
            self._presets: dict = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    #  Вычисление euler захвата                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_euler(tilt_deg: float, yaw_deg: float) -> list[float]:
        """
        Вычислить ориентацию gripper (ZYX Эйлер) из tilt и yaw.

        Tilt выполняется вокруг оси, перпендикулярной yaw-направлению детали.
        Это позволяет gripper'у наклоняться "к стенке" детали в зависимости
        от её ориентации на конвейере.

        Пример:
          yaw_deg=0, tilt=30° → gripper наклоняется по оси Y на 30°
          yaw_deg=90°, tilt=30° → gripper наклоняется по оси X на 30°

        Args:
            tilt_deg: наклон от вертикали (градусы, 0 = прямо вниз)
            yaw_deg:  ориентация детали в плоскости XY (градусы)

        Returns:
            [alpha, beta, gamma] в радианах (CoppeliaSim ZYX Эйлер)
        """
        tilt_rad = math.radians(tilt_deg)
        yaw_rad  = math.radians(yaw_deg)

        # Tilt выполняется перпендикулярно yaw-оси:
        #   бета-компонента (pitch) = tilt * cos(yaw + 90°) = -tilt * sin(yaw)
        #   альфа-компонента (roll) = tilt * sin(yaw + 90°) =  tilt * cos(yaw)
        #
        # При tilt=0 ориентация = [0, 0, yaw_rad] (gripper смотрит вниз, повёрнут по yaw)
        alpha = tilt_rad * math.cos(yaw_rad)      # крен
        beta  = -tilt_rad * math.sin(yaw_rad)     # тангаж

        return [alpha, beta, yaw_rad]

    @staticmethod
    def _apply_contact_offset(
        part_x: float,
        part_y: float,
        contact_offset: list[float],
        yaw_deg: float,
    ) -> tuple[float, float]:
        """
        Применить contact_offset с учётом yaw детали.

        [dx, dy] из пресета задаёт смещение в системе координат детали:
          dx — вдоль оси yaw детали (направление "вперёд")
          dy — перпендикулярно (направление "вправо")

        Поворачиваем этот вектор на yaw_rad, чтобы перейти в мировую СК.

        Args:
            part_x, part_y:   координаты центра детали (мир, метры)
            contact_offset:   [dx, dy] в СК детали (метры)
            yaw_deg:          ориентация детали (градусы)

        Returns:
            (grasp_x, grasp_y) — скорректированная точка захвата
        """
        dx, dy = contact_offset
        yaw_rad = math.radians(yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Поворот вектора [dx, dy] на yaw_rad
        world_dx = dx * cos_y - dy * sin_y
        world_dy = dx * sin_y + dy * cos_y

        return part_x + world_dx, part_y + world_dy

    # ------------------------------------------------------------------ #
    #  Основной метод                                                     #
    # ------------------------------------------------------------------ #

    def resolve(
        self,
        class_id: int,
        part_x: float,
        part_y: float,
        height_m: float,
        yaw_deg: float,
        belt_z: float,
    ) -> GraspParams:
        """
        Вычислить параметры захвата.

        Алгоритм:
          1. class_id → class_name → список пресетов из YAML
          2. Первый пресет с height_min ≤ height_m ≤ height_max
          3. Вычислить euler из tilt_deg и yaw_deg
          4. Применить contact_offset с учётом yaw
          5. pos.z = belt_z + height_m - GRIPPER_STANDOFF

        Args:
            class_id:  предсказанный класс детали (0=gaika, 1=vilka, 2=vtulka)
            part_x:    X-координата центра bbox детали (мир, метры)
            part_y:    Y-координата центра bbox детали (мир, метры)
            height_m:  высота верхней грани над лентой (из DepthEstimator, метры)
            yaw_deg:   ориентация детали в плоскости XY (из yaw-регрессора, градусы)
            belt_z:    Z-координата поверхности ленты (мировая СК, метры)

        Returns:
            GraspParams с pos и euler для RobotController.pick_part()
        """
        class_name = self.CLASS_NAMES.get(class_id, '')
        presets = self._presets.get(class_name, [])

        # Найти подходящий пресет по высоте
        matched = None
        for preset in presets:
            if preset['height_min'] <= height_m <= preset['height_max']:
                matched = preset
                break

        # Fallback: нет пресета → захват прямо вниз без смещения
        if matched is None:
            euler = [0.0, 0.0, math.radians(yaw_deg)]
            grasp_z = belt_z + height_m + self.GRIPPER_STANDOFF
            return GraspParams(
                pos=[part_x, part_y, grasp_z],
                euler=euler,
                preset_name='fallback',
            )

        # Ориентация gripper
        euler = self._compute_euler(matched['tilt_deg'], yaw_deg)

        # Точка контакта с учётом смещения по телу детали
        grasp_x, grasp_y = self._apply_contact_offset(
            part_x, part_y,
            matched.get('contact_offset', [0.0, 0.0]),
            yaw_deg,
        )

        # Высота подхода: поверхность ленты + высота детали + небольшой отступ
        grasp_z = belt_z + height_m + self.GRIPPER_STANDOFF

        return GraspParams(
            pos=[grasp_x, grasp_y, grasp_z],
            euler=euler,
            preset_name=matched.get('description', ''),
        )
