"""
Автоматическая аннотация: проекция 3D → 2D через матрицу камеры.

Алгоритм:
  1. Для каждой детали задан набор вершин в локальных координатах (упрощённая 3D-модель).
  2. Читается реальная матрица позы объекта из CoppeliaSim (sim.getObjectMatrix).
  3. Все локальные вершины трансформируются в мировые координаты.
  4. Вершины проецируются через pinhole-модель Vision Sensor → пиксели.
  5. min/max проекций → YOLO bbox.
  6. Yaw = третий угол Эйлера детали (вращение вокруг Z мировой оси).

Ключевое преимущество подхода: bbox корректен при ЛЮБОЙ ориентации детали,
потому что используется реальная матрица позы, а не допущение об упрощённой
ориентации (цилиндр вертикально и т.п.).

Формат YOLO (labels/):
  <class_id> <x_center> <y_center> <width> <height>
  все значения нормализованы [0, 1]

Формат yaw (yaw_labels/):
  <class_id> <yaw_degrees>   (yaw ∈ [0, 180) — угол по плоскости детали)
"""

import math
import numpy as np
from typing import Optional


# ------------------------------------------------------------------ #
#  Вспомогательные функции генерации вершин локальных моделей         #
# ------------------------------------------------------------------ #

def _cylinder_vertices(radius: float, half_height: float, n: int = 16) -> list[np.ndarray]:
    """
    Вершины цилиндра в локальных координатах (ось Z = ось цилиндра).

    Args:
        radius:      радиус цилиндра (м)
        half_height: половина высоты (м)
        n:           число точек по окружности

    Returns:
        список 3D-точек [x, y, z]
    """
    verts = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append(np.array([x, y, -half_height]))  # нижнее кольцо
        verts.append(np.array([x, y,  half_height]))  # верхнее кольцо
    return verts


def _hex_cylinder_vertices(
    r_collar: float,
    r_hex: float,
    h_total: float,
    n_hex: int = 6,
    n_circle: int = 12,
) -> list[np.ndarray]:
    """
    Вершины гайки шестигранной с буртиком в локальных координатах.

    Модель: буртик (цилиндр r_collar внизу) + шестигранник (r_hex сверху).
    Ось Z направлена вдоль оси гайки. Z=0 — нижний торец буртика.

    Args:
        r_collar:   радиус буртика (м)
        r_hex:      описанная окружность шестигранника (м)
        h_total:    полная высота гайки (м)
        n_hex:      число граней шестигранника
        n_circle:   число точек для окружности буртика

    Returns:
        список 3D-точек (вершины верхних/нижних контуров)
    """
    h_half = h_total / 2.0
    verts = []

    # Буртик — нижний и верхний контур (окружность)
    for i in range(n_circle):
        angle = 2 * math.pi * i / n_circle
        x = r_collar * math.cos(angle)
        y = r_collar * math.sin(angle)
        verts.append(np.array([x, y, -h_half]))           # нижний торец буртика
        verts.append(np.array([x, y, -h_half * 0.4]))     # верхний торец буртика

    # Шестигранник — нижний и верхний контур
    for i in range(n_hex):
        angle = 2 * math.pi * i / n_hex + math.pi / 6    # ориентация "плоскостью вверх"
        x = r_hex * math.cos(angle)
        y = r_hex * math.sin(angle)
        verts.append(np.array([x, y, -h_half * 0.4]))    # нижний торец шестигранника
        verts.append(np.array([x, y,  h_half]))           # верхний торец

    return verts


def _box_vertices(half_B: float, half_S: float, half_H: float) -> list[np.ndarray]:
    """
    Вершины прямоугольного параллелепипеда (8 углов).

    Используется для вилки: B — ширина вилки, S — толщина, H — высота.
    Ось Z направлена вдоль высоты. Z=0 — центр.

    Args:
        half_B: половина ширины по X (м)
        half_S: половина толщины по Y (м)
        half_H: половина высоты по Z (м)

    Returns:
        список 8 точек — углы параллелепипеда
    """
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append(np.array([sx * half_B, sy * half_S, sz * half_H]))
    return verts


def _cylinder_cap_vertices(
    r_body: float,
    r_flange: float,
    h_total: float,
    n: int = 16,
) -> list[np.ndarray]:
    """
    Вершины втулки (цилиндр с фланцем) в локальных координатах.

    Модель: тонкий широкий фланец внизу + узкий высокий цилиндр сверху.
    Ось Z = ось втулки, Z=0 — нижний торец фланца.

    Args:
        r_body:   радиус тела втулки (м)
        r_flange: радиус фланца (м)
        h_total:  полная высота втулки вместе с фланцем (м)
        n:        число точек по окружности

    Returns:
        список 3D-точек
    """
    h_half = h_total / 2.0
    h_flange = h_total * 0.15  # фланец = 15% высоты
    verts = []

    # Фланец — широкий нижний диск
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = r_flange * math.cos(angle)
        y = r_flange * math.sin(angle)
        verts.append(np.array([x, y, -h_half]))               # нижний торец
        verts.append(np.array([x, y, -h_half + h_flange]))    # верхний торец фланца

    # Тело втулки — узкий высокий цилиндр
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = r_body * math.cos(angle)
        y = r_body * math.sin(angle)
        verts.append(np.array([x, y, -h_half + h_flange]))    # основание тела
        verts.append(np.array([x, y,  h_half]))                # верхний торец

    return verts


# ------------------------------------------------------------------ #
#  Таблица локальных вершин для каждого класса                        #
# ------------------------------------------------------------------ #
#
# Размеры в метрах, соответствуют масштабу ×2 в CoppeliaSim:
#   gaika (0): ГОСТ 8918-69 × 2 → S=34мм, H=30мм, буртик⌀44мм
#   vilka (1): ГОСТ 12470-67 × 2 → B=64мм, S=50мм, H≈76мм
#   vtulka(2): цилиндр с фланцем → ⌀30мм тело, ⌀60мм фланец, H≈60мм
#
# Локальная система координат: центр объекта, ось Z вертикальна.
# НАСТРОИТЬ: уточнить по реальным размерам шаблонов в сцене.

PART_LOCAL_VERTICES: dict[int, list[np.ndarray]] = {
    0: _hex_cylinder_vertices(
        r_collar=0.022,   # радиус буртика 44мм/2
        r_hex=0.017,      # описанная окружность шестигранника S=34мм → r≈19.6мм → ≈0.017
        h_total=0.030,    # полная высота 30мм
    ),
    1: _box_vertices(
        half_B=0.032,   # ширина вилки B=64мм/2
        half_S=0.025,   # толщина S=50мм/2
        half_H=0.038,   # высота H≈76мм/2
    ),
    2: _cylinder_cap_vertices(
        r_body=0.015,    # радиус тела втулки ~30мм/2
        r_flange=0.030,  # радиус фланца ~60мм/2
        h_total=0.060,   # полная высота ~60мм
    ),
}


# ------------------------------------------------------------------ #
#  Annotator                                                           #
# ------------------------------------------------------------------ #

class Annotator:
    """
    Аннотатор: проецирует 3D bbox детали в 2D пиксели Vision Sensor.

    Использует vertex-transform подход: набор вершин упрощённой 3D-модели
    трансформируется через реальную матрицу позы объекта (sim.getObjectMatrix).
    Это обеспечивает корректный bbox при ЛЮБОЙ ориентации детали.
    """

    def __init__(self, sim, vision_handle: int, img_w: int = 640, img_h: int = 640) -> None:
        """
        Args:
            sim:           Remote API объект
            vision_handle: handle Vision Sensor в сцене
            img_w:         ширина изображения (пикселей)
            img_h:         высота изображения (пикселей)
        """
        self.sim = sim
        self.vision_handle = vision_handle
        self.img_w = img_w
        self.img_h = img_h

        # Угол обзора Vision Sensor (радианы)
        # Читаем из свойств сенсора; если не удалось — 30° по умолчанию
        try:
            fov_rad = sim.getObjectFloatParam(
                vision_handle,
                sim.visionfloatparam_perspective_angle
            )
        except Exception:
            fov_rad = math.radians(30.0)

        # Фокусное расстояние в пикселях (pinhole модель)
        # f = (W/2) / tan(FOV/2)
        self.fx: float = (img_w / 2.0) / math.tan(fov_rad / 2.0)
        self.fy: float = (img_h / 2.0) / math.tan(fov_rad / 2.0)
        self.cx: float = img_w / 2.0
        self.cy: float = img_h / 2.0

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы                                             #
    # ------------------------------------------------------------------ #

    def _get_camera_matrix_inv(self) -> np.ndarray:
        """
        Получить матрицу преобразования мир → камера (3×4).

        CoppeliaSim возвращает getObjectMatrix как список 12 элементов
        (строки 3×4 матрицы аффинного преобразования камера→мир).
        Нам нужна обратная: мир→камера.
        """
        # 3×4 матрица камера→мир: [R|T]
        m = self.sim.getObjectMatrix(self.vision_handle, -1)
        M = np.array(m).reshape(3, 4)
        R = M[:, :3]   # 3×3 ротация (столбцы = оси камеры в мировых координатах)
        T = M[:, 3]    # 3×1 позиция камеры в мировой СК

        # Матрица мир→камера: [R^T | -R^T * T]
        Rt = R.T
        t_cam = -Rt @ T
        return np.hstack([Rt, t_cam.reshape(3, 1)])  # 3×4

    def _world_to_pixel(
        self, point_world: np.ndarray, M_inv: np.ndarray
    ) -> tuple[float, float]:
        """
        Спроецировать точку из мировой СК в пиксели изображения.

        Args:
            point_world: [x, y, z] в метрах (мировая СК)
            M_inv:       матрица мир→камера (3×4)

        Returns:
            (u, v) — пиксельные координаты (может быть за пределами кадра)
        """
        p_h = np.array([*point_world, 1.0])  # гомогенные координаты
        p_cam = M_inv @ p_h                  # [xc, yc, zc] в СК камеры

        # Точка должна быть перед камерой (zc > 0)
        if p_cam[2] <= 0:
            return float('inf'), float('inf')

        # Перспективная проекция (pinhole)
        # CoppeliaSim: ось X камеры → противоположна пиксельной U → инвертируем
        u = -self.fx * p_cam[0] / p_cam[2] + self.cx
        # CoppeliaSim: ось Y камеры направлена вверх → инвертируем v
        v = -self.fy * p_cam[1] / p_cam[2] + self.cy
        return u, v

    # ------------------------------------------------------------------ #
    #  Bbox из разности кадров (background subtraction)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def bbox_from_diff(
        frame: np.ndarray,
        background: np.ndarray,
        threshold: int = 18,
        padding: int = 3,
    ) -> Optional[tuple[int, int, int, int]]:
        """
        Вычислить пиксельный bbox детали по разности с фоновым кадром.

        Алгоритм:
          diff = |frame - background| по max каналу → бинарная маска
          → min/max строк и столбцов маски → (u_min, v_min, u_max, v_max)

        Преимущества перед AABB API:
          - Точность 100% при любой ориентации детали
          - Не зависит от CoppeliaSim API (getObjectFloatParam и пр.)
          - Работает для составных объектов (vilka = несколько shape)

        Args:
            frame:      кадр с деталью  (H, W, 3) uint8
            background: пустая лента    (H, W, 3) uint8, снятая ДО спавна
            threshold:  порог разности (0-255); 18 надёжно ловит детали
            padding:    расширить bbox на N пикселей с каждой стороны

        Returns:
            (u_min, v_min, u_max, v_max) в пикселях, или None если деталь не найдена
        """
        diff = np.abs(frame.astype(np.float32) - background.astype(np.float32))
        mask = diff.max(axis=2) > threshold  # (H, W) bool

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

        H, W = frame.shape[:2]
        c0 = max(0,     c0 - padding)
        r0 = max(0,     r0 - padding)
        c1 = min(W - 1, c1 + padding)
        r1 = min(H - 1, r1 + padding)

        return (c0, r0, c1, r1)

    def annotate_from_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
        part_handle: int,
        class_id: int,
    ) -> Optional[dict]:
        """
        Сформировать аннотацию по заранее вычисленному пиксельному bbox.

        Используется совместно с bbox_from_diff() вместо annotate().

        Args:
            bbox_px:     (u_min, v_min, u_max, v_max) в пикселях
            part_handle: handle детали (только для чтения yaw)
            class_id:    0=gaika, 1=vilka, 2=vtulka

        Returns:
            dict с полями 'yolo' и 'yaw', или None если bbox вырожден
        """
        u_min, v_min, u_max, v_max = bbox_px
        bw = u_max - u_min
        bh = v_max - v_min
        if bw <= 1 or bh <= 1:
            return None

        # YOLO-нормализация
        x_c = ((u_min + u_max) / 2.0) / self.img_w
        y_c = ((v_min + v_max) / 2.0) / self.img_h
        w   = bw / self.img_w
        h   = bh / self.img_h

        # Yaw — угол Эйлера вокруг Z из позы объекта
        euler   = self.sim.getObjectOrientation(part_handle, -1)
        yaw_rad = euler[2]
        yaw_deg = math.degrees(yaw_rad) % 360.0

        symmetry: dict[int, float] = {0: 60.0, 1: 180.0, 2: 360.0}
        yaw_deg = yaw_deg % symmetry.get(class_id, 180.0)

        return {
            'yolo': (class_id, x_c, y_c, w, h),
            'yaw':  yaw_deg,
        }

    def _get_shape_aabb_world(
        self, obj_handle: int, M: np.ndarray
    ) -> list[np.ndarray]:
        """
        Вернуть 8 мировых вершин AABB shape-объекта.
        Возвращает [] если objbbox недоступен (dummy, joint и т.п.).

        getObjectFloatParam возвращает None (не исключение!) если параметр
        недоступен для данного типа объекта — поэтому проверяем явно.
        """
        sim = self.sim
        min_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_x)
        if min_x is None:
            return []
        min_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_y)
        min_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_z)
        max_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_x)
        max_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_y)
        max_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_z)
        if any(v is None for v in (min_y, min_z, max_x, max_y, max_z)):
            return []
        corners = []
        for x in (min_x, max_x):
            for y in (min_y, max_y):
                for z in (min_z, max_z):
                    corners.append(M @ np.array([x, y, z, 1.0]))
        return corners

    def _get_world_vertices(self, part_handle: int, class_id: int) -> list[np.ndarray]:
        """
        Получить угловые точки AABB детали (и всех дочерних shape) в мировой СК.

        Стратегия (порядок приоритета):
          1. modelbbox на корне — если объект помечен как «модель» в CoppeliaSim,
             возвращает суммарный AABB всех дочерних shape за один вызов.
             getObjectFloatParam возвращает None (не исключение!) если модель не задана.
          2. Итерация getObjectsInTree + objbbox на каждом shape —
             для немаркированных моделей или составных объектов.
             None-результаты (дummies, joints) отфильтровываются в _get_shape_aabb_world.

        Args:
            part_handle: handle корня детали в сцене
            class_id:    не используется (оставлен для совместимости)

        Returns:
            список точек в мировой СК
        """
        sim = self.sim

        # Матрица корень→мир (3×4)
        m = sim.getObjectMatrix(part_handle, -1)
        M_root = np.array(m).reshape(3, 4)

        # --- Попытка 1: modelbbox (суммарный AABB всей модели) ---
        # getObjectFloatParam возвращает None если объект не помечен как модель
        min_x = sim.getObjectFloatParam(part_handle, sim.objfloatparam_modelbbox_min_x)
        if min_x is not None:
            min_y = sim.getObjectFloatParam(part_handle, sim.objfloatparam_modelbbox_min_y)
            min_z = sim.getObjectFloatParam(part_handle, sim.objfloatparam_modelbbox_min_z)
            max_x = sim.getObjectFloatParam(part_handle, sim.objfloatparam_modelbbox_max_x)
            max_y = sim.getObjectFloatParam(part_handle, sim.objfloatparam_modelbbox_max_y)
            max_z = sim.getObjectFloatParam(part_handle, sim.objfloatparam_modelbbox_max_z)
            if all(v is not None for v in (min_y, min_z, max_x, max_y, max_z)):
                return [
                    M_root @ np.array([x, y, z, 1.0])
                    for x in (min_x, max_x)
                    for y in (min_y, max_y)
                    for z in (min_z, max_z)
                ]

        # --- Попытка 2: итерация по всем объектам поддерева ---
        # objbbox каждого shape в его СОБСТВЕННОЙ мировой матрице
        try:
            all_objects = sim.getObjectsInTree(part_handle, sim.handle_all, 0)
        except Exception:
            all_objects = []
        if not all_objects:
            all_objects = [part_handle]

        world_corners: list[np.ndarray] = []
        for obj_handle in all_objects:
            try:
                m_obj = sim.getObjectMatrix(obj_handle, -1)
                M_obj = np.array(m_obj).reshape(3, 4)
                world_corners.extend(self._get_shape_aabb_world(obj_handle, M_obj))
            except Exception:
                continue

        return world_corners

    # ------------------------------------------------------------------ #
    #  Основной метод аннотации                                           #
    # ------------------------------------------------------------------ #

    def annotate(
        self, part_handle: int, class_id: int
    ) -> Optional[dict]:
        """
        Вычислить аннотацию детали в текущей позе.

        Алгоритм:
          1. Получить мировые координаты вершин модели (через матрицу позы).
          2. Спроецировать все вершины в пиксели камеры.
          3. min/max → bbox в пикселях → нормализовать в [0,1].
          4. Читать yaw как третий угол Эйлера (ось Z).

        Args:
            part_handle: handle детали в сцене
            class_id:    0=gaika, 1=vilka, 2=vtulka

        Returns:
            dict с полями:
              'yolo': (class_id, x_c, y_c, w, h) нормализованные [0,1]
              'yaw':  угол yaw в градусах (с учётом симметрии детали)
            None если bbox выходит за пределы изображения
        """
        M_inv = self._get_camera_matrix_inv()
        world_verts = self._get_world_vertices(part_handle, class_id)  # AABB из CoppeliaSim

        if not world_verts:
            return None

        # Проецируем все вершины → собираем пиксельные координаты
        us, vs = [], []
        for v_world in world_verts:
            u, v = self._world_to_pixel(v_world, M_inv)
            if not (math.isinf(u) or math.isinf(v)):
                us.append(u)
                vs.append(v)

        if not us:
            return None

        u_min, u_max = min(us), max(us)
        v_min, v_max = min(vs), max(vs)

        # Проверка: bbox хотя бы частично внутри кадра
        if u_max < 0 or u_min > self.img_w or v_max < 0 or v_min > self.img_h:
            return None

        # Обрезать по границам кадра
        u_min_clip = max(0.0, u_min)
        u_max_clip = min(float(self.img_w), u_max)
        v_min_clip = max(0.0, v_min)
        v_max_clip = min(float(self.img_h), v_max)

        # Пропустить если менее 50% bbox внутри кадра (объект почти за кадром)
        area_full = (u_max - u_min) * (v_max - v_min)
        area_clip = (u_max_clip - u_min_clip) * (v_max_clip - v_min_clip)
        if area_full <= 0 or area_clip / area_full < 0.5:
            return None

        u_min, u_max = u_min_clip, u_max_clip
        v_min, v_max = v_min_clip, v_max_clip

        # YOLO формат: нормализованные x_center, y_center, width, height
        x_c = ((u_min + u_max) / 2.0) / self.img_w
        y_c = ((v_min + v_max) / 2.0) / self.img_h
        w   = (u_max - u_min) / self.img_w
        h   = (v_max - v_min) / self.img_h

        # Yaw: третий угол Эйлера (вращение вокруг Z мировой оси)
        euler = self.sim.getObjectOrientation(part_handle, -1)
        yaw_rad = euler[2]
        yaw_deg = math.degrees(yaw_rad) % 360.0

        # Нормализация yaw по симметрии детали:
        #   gaika (0): шестигранник — симметрия 60°
        #   vilka (1): U-форма — симметрия 180°
        #   vtulka (2): цилиндр — симметрия 360° (полная, любой yaw эквивалентен)
        symmetry: dict[int, float] = {0: 60.0, 1: 180.0, 2: 360.0}
        sym = symmetry.get(class_id, 180.0)
        yaw_deg = yaw_deg % sym

        return {
            'yolo': (class_id, x_c, y_c, w, h),
            'yaw':  yaw_deg,
        }

    # ------------------------------------------------------------------ #
    #  Сохранение аннотаций на диск                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_yolo_label(path: str, annotations: list[tuple]) -> None:
        """
        Сохранить YOLO-аннотацию в txt файл.

        Args:
            path:        полный путь к файлу (с .txt)
            annotations: список кортежей (class_id, x_c, y_c, w, h)
        """
        with open(path, 'w') as f:
            for ann in annotations:
                class_id, x_c, y_c, w, h = ann
                f.write(f'{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n')

    @staticmethod
    def save_yaw_label(path: str, class_id: int, yaw_deg: float) -> None:
        """
        Сохранить yaw-аннотацию в txt файл.

        Args:
            path:      полный путь к файлу
            class_id:  класс детали
            yaw_deg:   нормализованный yaw в градусах
        """
        with open(path, 'w') as f:
            f.write(f'{class_id} {yaw_deg:.4f}\n')
