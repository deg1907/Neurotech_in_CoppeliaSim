"""
Управление конвейером и спавном деталей.

Конвейер: generic conveyor belt из Model Browser CoppeliaSim.
Управление скоростью: sim.setBufferProperty с таблицей {vel: ...}

Создание деталей: копирование шаблонов из сцены (sim.copyPasteObjects).
  Шаблоны /gaika, /vilka, /vtulka должны присутствовать в сцене — они
  невидимы/статичны и служат только источниками для копирования.
  Копия появляется в spawn_point и едет по конвейеру.
"""

from typing import Optional
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# RGB-цвета деталей — применяются к копии после спавна
PART_COLORS: dict[int, list[float]] = {
    0: [0.70, 0.70, 0.75],  # gaika  — сталь/серебро
    1: [0.65, 0.65, 0.70],  # vilka  — тёмная сталь
    2: [0.80, 0.82, 0.85],  # vtulka — алюминий
}

CLASS_NAMES = {0: 'gaika', 1: 'vilka', 2: 'vtulka'}


class ConveyorController:
    """Управление конвейером и подачей деталей."""

    # НАСТРОИТЬ: путь к корневому объекту конвейера в сцене
    CONVEYOR_PATH: str = '/conveyor'  # уточнить по своей сцене

    # Скорость ленты в м/с
    DEFAULT_VEL: float = 0.08

    def __init__(self, sim) -> None:
        """
        Args:
            sim: объект Remote API (sim = client.require('sim'))
        """
        self.sim = sim

        # Handle конвейера (корневой объект модели из библиотеки)
        self.conveyor_handle: int = sim.getObject(self.CONVEYOR_PATH)

        # Proximity sensors конвейера (добавлены вручную в сцену)
        # НАСТРОИТЬ: уточни имена если отличаются
        self.arrival_sensor_handle: int  = sim.getObject('/conveyor/conveyor_sensor_arrival')
        self.pickup_sensor_handle: int   = sim.getObject('/conveyor/conveyor_sensor_pickup')

        # Spawn point — точка появления детали
        self.spawn_handle: int = sim.getObject('/spawn_point')

        # Счётчик созданных деталей (для уникальных имён)
        self._part_counter: int = 0

    # ------------------------------------------------------------------ #
    #  Управление скоростью                                                #
    # ------------------------------------------------------------------ #

    def set_velocity(self, vel: float) -> None:
        """
        Установить скорость ленты конвейера.

        Args:
            vel: скорость в м/с (0.0 = стоп)
        """
        self.sim.setBufferProperty(
            self.conveyor_handle,
            'customData.__ctrl__',
            self.sim.packTable({'vel': vel})
        )

    def start(self, vel: Optional[float] = None) -> None:
        """Запустить конвейер."""
        self.set_velocity(vel if vel is not None else self.DEFAULT_VEL)

    def stop(self) -> None:
        """Остановить конвейер."""
        self.set_velocity(0.0)

    def get_state(self) -> dict:
        """Прочитать текущее состояние конвейера (скорость, позиция и т.д.)."""
        return self.sim.readCustomTableData(self.conveyor_handle, '__state__') or {}

    # ------------------------------------------------------------------ #
    #  Сенсоры                                                             #
    # ------------------------------------------------------------------ #

    def check_arrival(self) -> tuple[bool, Optional[int]]:
        """
        Проверить сенсор прибытия (деталь приближается к зоне захвата).

        Returns:
            (detected: bool, part_handle: int | None)
        """
        result, _, detected_handle, *_ = self.sim.readProximitySensor(
            self.arrival_sensor_handle
        )
        #print('arrival: ', result, detected_handle)
        if result == 1:
            return True, detected_handle
        return False, None

    def check_pickup_ready(self) -> tuple[bool, Optional[int]]:
        """
        Проверить сенсор зоны захвата (деталь точно в позиции pickup).

        Returns:
            (ready: bool, part_handle: int | None)
        """
        result, _, detected_handle, *_ = self.sim.readProximitySensor(
            self.pickup_sensor_handle
        )
        #print('pickup: ', result, detected_handle)
        if result == 1:
            return True, detected_handle
        return False, None

    def wait_for_part(self, max_steps: int = 3000) -> tuple[bool, Optional[int]]:
        """
        Ждать пока деталь не окажется в зоне захвата.
        Автоматически замедляет конвейер при приближении.

        Args:
            max_steps: максимальное количество шагов симуляции

        Returns:
            (success: bool, part_handle: int | None)
        """
        self.start()
        for _ in range(max_steps):
            self.sim.step()

            # При приближении — замедлить для точной остановки
            arrival, _ = self.check_arrival()
            if arrival:
                self.set_velocity(0.02)

            # Деталь в зоне захвата — стоп
            ready, handle = self.check_pickup_ready()
            if ready:
                self.stop()
                # Дать детали успокоиться
                for _ in range(30):
                    self.sim.step()
                return True, handle

        self.stop()
        return False, None

    # ------------------------------------------------------------------ #
    #  Создание и удаление деталей                                         #
    # ------------------------------------------------------------------ #
    def spawn_part(self, class_id: int) -> int:
        """
        Создать новую деталь в spawn_point.

        Деталь состоит из примитивов согласно PART_SHAPES.
        Если у класса несколько примитивов — объединяются в одну фигуру.

        Args:
            class_id: 0=bolt, 1=nut, 2=washer

        Returns:
            handle созданной детали
        """
        sim = self.sim
        handles = []
        template_paths = {0: '/gaika', 1: '/vilka', 2: '/vtulka'}
        if class_id not in template_paths:
            raise ValueError(f'Неизвестный тип детали: {class_id}')
        template_handle = sim.getObject(template_paths[class_id])
        h = sim.copyPasteObjects([template_handle], 0)

        # Позиция примитива относительно spawn_point
        spawn_pos = sim.getObjectPosition(self.spawn_handle, -1)
        sim.setObjectPosition(h[0], -1, [
            spawn_pos[0],
            spawn_pos[1],
            spawn_pos[2],
        ])

        # Цвет
        color = PART_COLORS[class_id]
        sim.setShapeColor(h[0], None, sim.colorcomponent_ambient_diffuse, color)

        handles.append(h[0])

        part_handle = handles[0]
        # Уникальное имя
        name = f'part_{CLASS_NAMES[class_id]}_{self._part_counter:03d}'
        sim.setObjectAlias(part_handle, name)
        self._part_counter += 1


        # Случайная 3D-ориентация — деталь падает и ложится на произвольную грань
        roll  = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)
        yaw   = np.random.uniform(0, 2 * np.pi)
        sim.setObjectOrientation(part_handle, -1, [roll, pitch, yaw])
        return part_handle

    def remove_part(self, part_handle: int) -> None:
        """Удалить деталь из сцены (после успешной сортировки)."""
        self.sim.removeObject(part_handle)
