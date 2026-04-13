"""
Управление манипулятором ABB IRB 140 через ZMQ Remote API.

IK решается автоматически скриптом сцены (/IRB140/Script) на каждом шаге
симуляции. Python только перемещает dummy target — скрипт IRB140 сам
вычисляет углы суставов.

Захват: BaxterVacuumCupWithGUI — управляется через Int32 сигнал
  активация:   sim.setInt32Signal(vacuum_signal, 1)
  деактивация: sim.setInt32Signal(vacuum_signal, 0)
  Когда сигнал = 1, Lua-скрипт захвата автоматически ищет ближайший
  respondable shape и соединяет его через loop closure dummy link.

Режимы pick_part():
  - grasp_params передан (из GraspPlanner) — использует pos и euler из пресета.
    Поддерживает захват с наклоном для сложных поз детали.
  - grasp_params=None — fallback: вертикальный захват с GRASP_Z_OFFSET[class_id].
"""

from typing import Optional
import sys
from pathlib import Path
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# GraspParams импортируется опционально — модуль может работать и без планировщика
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from grasp_planner import GraspParams
except ImportError:
    GraspParams = None  # type: ignore


# ------------------------------------------------------------------ #
#  Вспомогательная функция поиска объекта по имени в поддереве        #
# ------------------------------------------------------------------ #

def _find_in_subtree(sim, root_handle: int, alias: str) -> int:
    """
    Найти объект с именем alias внутри поддерева root_handle.

    Args:
        sim:         Remote API объект
        root_handle: handle корневого объекта поиска (-1 = вся сцена)
        alias:       имя объекта (sim.getObjectAlias без флагов)

    Returns:
        handle найденного объекта

    Raises:
        RuntimeError если объект не найден
    """
    objects = sim.getObjectsInTree(root_handle, sim.handle_all, 0)
    for h in objects:
        if sim.getObjectAlias(h, 0) == alias:
            return h
    scope = f'handle={root_handle}' if root_handle != -1 else 'сцена'
    raise RuntimeError(f'Объект "{alias}" не найден ({scope})')


class RobotController:
    """Контроллер манипулятора IRB 140 с вакуумным захватом BaxterVacuumCup."""

    # Позиции сброса в бин [x, y, z] в метрах (мировая СК).
    # Z — высота сброса (над верхним краем бина).
    # НАСТРОИТЬ: подогнать по реальному расположению бинов в сцене.
    BIN_POSITIONS: dict[int, list[float]] = {
        0: [-0.45,  0.20, 0.40],  # gaika
        1: [-0.45, -0.20, 0.40],  # vilka
        2: [-0.20, -0.45, 0.40],  # vtulka
    }

    # Безопасная домашняя позиция (высоко над рабочей зоной)
    HOME_POS: list[float] = [0.0, -0.5, 0.65]

    # Позиция над конвейером перед опусканием к детали
    ABOVE_PICKUP_POS: list[float] = [0.50, 0.0, 0.40]

    # Ориентация захвата: смотрит вниз (-Z мира)
    GRASP_EULER: list[float] = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------ #
    #  Fallback-параметры захвата (используются если GraspPlanner недоступен)
    # ------------------------------------------------------------------ #
    # Основная логика — через GraspPlanner.resolve() + grasp_presets.yaml.
    # Эти константы применяются только при grasp_params=None в pick_part().

    # Дополнительная высота захвата по классу детали (м), прибавляется к belt_surface_z.
    GRASP_Z_OFFSET: dict[int, float] = {
        0: 0.030,   # gaika  — высота ~30мм (наиболее частая поза: буртиком вниз)
        1: 0.038,   # vilka  — высота ~76мм/2 (наиболее частая поза: основанием вниз)
        2: 0.060,   # vtulka — высота ~60мм (вертикально)
    }

    # Смещение точки захвата от центра bbox (dx, dy) в метрах.
    # НАСТРОИТЬ: подобрать эмпирически для наиболее частой позы каждой детали.
    GRASP_XY_OFFSET: dict[int, list[float]] = {
        0: [0.0, 0.0],   # gaika  — центр
        1: [0.0, 0.0],   # vilka  — центр (fallback без учёта позы)
        2: [0.0, 0.0],   # vtulka — центр
    }

    def __init__(self, sim) -> None:
        """
        Инициализация: поиск всех нужных объектов в сцене.

        Args:
            sim: объект Remote API (sim = client.require('sim'))
        """
        self.sim = sim
        irb140 = sim.getObject('/IRB140')

        # IK target (внутри manipulationSphere) и tip (в конце цепи)
        # Ищем по имени в поддереве IRB140 — не зависит от глубины иерархии
        self.target_handle: int = _find_in_subtree(sim, irb140, 'target')
        self.tip_handle:    int = _find_in_subtree(sim, irb140, 'tip')

        # BaxterVacuumCupWithGUI — захват
        self.gripper_handle: int = _find_in_subtree(sim, irb140, 'BaxterVacuumCupWhithGUI')

        # Сигнал управления вакуумом: вычисляется так же, как в Lua-скрипте захвата
        # sim.getObjectAlias(b, 4) → 'BaxterVacuumCupWhithGUI__NNN__'
        vacuum_alias = sim.getObjectAlias(self.gripper_handle, 4)
        self.vacuum_signal: str = vacuum_alias + '_active'

        # loopClosureDummy1 — по нему определяем факт захвата детали
        # Когда деталь захвачена: его parent = захваченная деталь (не gripper)
        self.lcd1_handle: int = _find_in_subtree(
            sim, self.gripper_handle, 'loopClosureDummy1'
        )

    # ------------------------------------------------------------------ #
    #  Низкоуровневое управление                                           #
    # ------------------------------------------------------------------ #

    def set_target_pose(
        self,
        pos: list[float],
        euler: Optional[list[float]] = None,
    ) -> None:
        """
        Переместить IK target (dummy, который тянет за собой манипулятор).

        Args:
            pos:   [x, y, z] в метрах, мировая СК
            euler: [alpha, beta, gamma] в радианах (опционально)
        """
        self.sim.setObjectPosition(self.target_handle, -1, pos)
        if euler is not None:
            self.sim.setObjectOrientation(self.target_handle, -1, euler)

    def wait_convergence(
        self,
        tol: float = 0.009,
        max_steps: int = 600,
    ) -> bool:
        """
        Ждать пока TCP (tip) не достигнет target с точностью tol метров.

        Args:
            tol:       допустимое расстояние tip–target, м
            max_steps: максимальное количество шагов симуляции до timeout

        Returns:
            True если сошлось, False если timeout
        """
        for _ in range(max_steps):
            self.sim.step()
            tip_pos = np.array(self.sim.getObjectPosition(self.tip_handle, -1))
            tgt_pos = np.array(self.sim.getObjectPosition(self.target_handle, -1))
            if np.linalg.norm(tip_pos - tgt_pos) < tol:
                return True
        return False

    def move_to(
        self,
        pos: list[float],
        euler: Optional[list[float]] = None,
        tol: float = 0.003,
    ) -> bool:
        """
        Переместить манипулятор в позицию и ждать прибытия.

        Args:
            pos:   целевая позиция [x, y, z] в метрах
            euler: целевая ориентация [a, b, g] в радианах
            tol:   точность позиционирования, м

        Returns:
            True если достигнуто, False если timeout
        """
        self.set_target_pose(pos, euler)
        return self.wait_convergence(tol=tol)

    def go_home(self) -> bool:
        """Вернуться в безопасную домашнюю позицию."""
        return self.move_to(self.HOME_POS)

    # ------------------------------------------------------------------ #
    #  Вакуумный захват                                                    #
    # ------------------------------------------------------------------ #

    def vacuum_on(self) -> None:
        """
        Активировать вакуумный захват.
        Lua-скрипт начнёт искать ближайший respondable shape у сенсора
        и создаст loop closure link при обнаружении.
        """
        self.sim.setInt32Signal(self.vacuum_signal, 1)

    def vacuum_off(self) -> None:
        """
        Деактивировать вакуумный захват.
        Lua-скрипт разорвёт loop closure link и отпустит деталь.
        """
        self.sim.setInt32Signal(self.vacuum_signal, 0)

    def has_part(self) -> bool:
        """
        Проверить, захвачена ли деталь.

        Когда деталь захвачена, loopClosureDummy1 становится child
        захваченного объекта (а не child gripper'а).
        """
        parent = self.sim.getObjectParent(self.lcd1_handle)
        return parent != self.gripper_handle

    def wait_for_grab(self, max_steps: int = 100) -> bool:
        """
        Ждать пока деталь не будет захвачена (loop closure установлен).

        Args:
            max_steps: максимальное количество шагов

        Returns:
            True если захвачено
        """
        for _ in range(max_steps):
            self.sim.step()
            if self.has_part():
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Высокоуровневые операции pick & place                               #
    # ------------------------------------------------------------------ #

    def pick_part(
        self,
        part_x: float,
        part_y: float,
        class_id: int = -1,
        grasp_params=None,
    ) -> bool:
        """
        Полный цикл захвата детали с конвейера.

        Два режима работы:
          - grasp_params передан (GraspParams из GraspPlanner):
              использует pos и euler из пресета захвата.
              Поддерживает наклонный захват для сложных поз.
          - grasp_params=None:
              fallback — вертикальный захват с GRASP_Z_OFFSET[class_id].

        Последовательность:
          1. Переместиться над точкой захвата (безопасная высота)
          2. Активировать вакуум
          3. Опуститься к точке захвата с нужной ориентацией
          4. Ждать захвата (loop closure)
          5. Подняться в ABOVE_PICKUP_POS

        Args:
            part_x:       X-координата центра bbox детали (мир, метры)
            part_y:       Y-координата центра bbox детали (мир, метры)
            class_id:     класс детали (0=gaika, 1=vilka, 2=vtulka). -1 = неизвестен.
            grasp_params: GraspParams из GraspPlanner (приоритет над fallback).

        Returns:
            True если деталь успешно захвачена
        """
        # Высота верхней поверхности конвейерной ленты (м)
        belt_surface_z = 0.131

        # Безопасная высота над рабочей зоной перед опусканием
        above_z = belt_surface_z + 0.15

        if grasp_params is not None:
            # ── Режим GraspPlanner ──────────────────────────────────────────
            # Используем pos и euler из пресета захвата
            grasp_x, grasp_y, grasp_z = grasp_params.pos
            grasp_euler = grasp_params.euler
        else:
            # ── Fallback: вертикальный захват ────────────────────────────────
            z_offset = self.GRASP_Z_OFFSET.get(class_id, 0.0)
            grasp_z  = belt_surface_z + z_offset
            xy_off   = self.GRASP_XY_OFFSET.get(class_id, [0.0, 0.0])
            grasp_x  = part_x + xy_off[0]
            grasp_y  = part_y + xy_off[1]
            grasp_euler = self.GRASP_EULER

        # 1. Над точкой захвата (с нужным yaw, но без наклона — безопасный подъём)
        above_euler = [0.0, 0.0, grasp_euler[2]] if len(grasp_euler) == 3 else self.GRASP_EULER
        if not self.move_to([grasp_x, grasp_y, above_z], above_euler):
            return False

        # 2. Активировать вакуум
        self.vacuum_on()

        # 3. Опуститься к детали с полной ориентацией захвата (включая tilt)
        if not self.move_to([grasp_x, grasp_y, grasp_z], grasp_euler, tol=0.005):
            self.vacuum_off()
            self.move_to(self.ABOVE_PICKUP_POS, self.GRASP_EULER)
            return False

        # 4. Ждать захвата
        if not self.wait_for_grab(max_steps=150):
            self.vacuum_off()
            self.move_to(self.ABOVE_PICKUP_POS, self.GRASP_EULER)
            return False

        # 5. Подъём — сначала убрать наклон (в вертикаль), потом поднять
        self.move_to([grasp_x, grasp_y, above_z], above_euler)
        self.move_to(self.ABOVE_PICKUP_POS, self.GRASP_EULER)
        return True

    def place_to_bin(self, class_id: int) -> None:
        """
        Переместить удерживаемую деталь в бин и отпустить.

        Args:
            class_id: 0=bolt, 1=nut, 2=washer
        """
        drop_pos = self.BIN_POSITIONS.get(class_id, self.HOME_POS)  # 0=gaika,1=vilka,2=vtulka

        # Переместиться над бином
        self.move_to(drop_pos, self.GRASP_EULER)

        # Деактивировать вакуум — деталь отпадает под гравитацией
        self.vacuum_off()

        # Дать детали упасть в бин
        for _ in range(80):
            self.sim.step()
