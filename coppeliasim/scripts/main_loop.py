"""
Главный цикл сортировки деталей.

Запуск:
    1. Открыть CoppeliaSim, загрузить sorting_scene.ttt
    2. Добавить depth Vision Sensor (/vision_sensor_depth) в сцену
    3. Запустить симуляцию (Play)
    4. python main_loop.py [n_parts] [--weights path/to/detector_best.pt]

Режим: синхронный (client.setStepping(True)) — Python управляет каждым
шагом симуляции. Это гарантирует детерминированность и позволяет точно
синхронизировать захват кадров с движением конвейера/робота.

Pipeline одного цикла сортировки:
  1. Спавн детали в случайной 3D-ориентации
  2. Конвейер подаёт деталь в зону захвата
  3. RGB-кадр → Detector CNN → (class_id, bbox, yaw)
  4. Depth-кадр → DepthEstimator → height_m
  5. GraspPlanner.resolve() → GraspParams (pos, euler)
  6. RobotController.pick_part(grasp_params) → захват
  7. place_to_bin → сортировка

Зависимости: robot_control.py, conveyor.py, vision.py,
             depth_estimator.py, grasp_planner.py,
             integration/pipeline.py
"""

import sys
import argparse
import random
from pathlib import Path
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from robot_control import RobotController
from conveyor import ConveyorController
from vision import VisionSensor, VisionSensorDepth
from depth_estimator import DepthEstimator
from grasp_planner import GraspPlanner

# Путь к integration/pipeline.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'integration'))
from pipeline import DetectionPipeline


# ------------------------------------------------------------------ #
#  Константы                                                           #
# ------------------------------------------------------------------ #

CLASS_NAMES = {0: 'gaika', 1: 'vilka', 2: 'vtulka'}

# Z-координата поверхности конвейерной ленты (м).
# НАСТРОИТЬ: уточнить по реальной сцене.
BELT_Z: float = 0.131


# ------------------------------------------------------------------ #
#  Статистика сортировки                                               #
# ------------------------------------------------------------------ #

class SortingStats:
    """Накапливает метрики сортировки для финального отчёта."""

    def __init__(self) -> None:
        self.total: int = 0
        self.correct: int = 0
        self.failed_picks: int = 0
        self.failed_detections: int = 0

    def record(self, success: bool) -> None:
        self.total += 1
        if success:
            self.correct += 1

    def print_summary(self) -> None:
        acc = self.correct / self.total * 100 if self.total > 0 else 0.0
        print(f'\n=== Итоги сортировки ===')
        print(f'Всего деталей:      {self.total}')
        print(f'Правильно:          {self.correct}')
        print(f'Точность:           {acc:.1f}%')
        print(f'Ошибок захвата:     {self.failed_picks}')
        print(f'Ошибок детекции:    {self.failed_detections}')


# ------------------------------------------------------------------ #
#  Главный цикл                                                        #
# ------------------------------------------------------------------ #

def main(n_parts: int = 20, weights_path: str = None) -> None:
    """
    Запустить сортировку n_parts деталей.

    Args:
        n_parts:      количество деталей для сортировки
        weights_path: путь к весам детектора (.pt)
    """
    if weights_path is None:
        # Поиск весов: сначала weights/, потом models/weights/
        root = Path(__file__).parent.parent.parent
        for candidate in [root / 'weights' / 'detector_best.pt',
                          root / 'models' / 'weights' / 'detector_best.pt']:
            if candidate.exists():
                weights_path = str(candidate)
                break
        if weights_path is None:
            raise FileNotFoundError(
                'Веса детектора не найдены. '
                'Укажи путь явно: python main_loop.py --weights path/to/detector_best.pt'
            )

    # Подключение к CoppeliaSim
    client = RemoteAPIClient()
    sim    = client.require('sim')

    # Синхронный режим
    client.setStepping(True)
    sim.startSimulation()

    # Инициализация компонентов
    robot        = RobotController(sim)
    conveyor     = ConveyorController(sim)
    camera       = VisionSensor(sim)
    depth_sensor = VisionSensorDepth(sim)
    depth_est    = DepthEstimator()
    grasp_planner = GraspPlanner()  # читает grasp_presets.yaml
    stats        = SortingStats()

    # Инициализация pipeline детекции
    vision_handle = sim.getObject('/vision_sensor_main')
    pipeline = DetectionPipeline(
        weights_path=weights_path,
        sim=sim,
        vision_handle=vision_handle,
        conf_thresh=0.3,   # 0.3 для недообученной модели, повысить до 0.5 после 50 эпох
        device='cpu',
    )

    # Калибровка depth-сенсора на пустом конвейере
    print('Калибровка depth-сенсора (пустой конвейер)...')
    for _ in range(10):
        sim.step()
    try:
        depth_frame_cal = depth_sensor.capture_depth()
        belt_depth = depth_est.calibrate_belt(depth_frame_cal)
        print(f'  Belt depth: {belt_depth:.3f} м')
    except Exception as e:
        print(f'  [WARN] Depth сенсор недоступен: {e}')
        print(f'  Используется дефолтная высота захвата из grasp_presets.yaml')
        depth_sensor = None

    print(f'\nЗапуск сортировки: {n_parts} деталей')
    robot.go_home()

    for i in range(n_parts):
        print(f'\n--- Деталь {i+1}/{n_parts} ---')

        # 1. Спавн новой детали (случайный класс)
        class_id_gt = random.randint(0, 2)
        part_handle = conveyor.spawn_part(class_id_gt)
        print(f'  Спавн: {CLASS_NAMES[class_id_gt]} (handle={part_handle})')

        # 2. Ждать пока деталь доедет до зоны захвата
        success, _ = conveyor.wait_for_part(max_steps=1000)
        if not success:
            print('  [ОШИБКА] Деталь не прибыла за отведённое время')
            conveyor.remove_part(part_handle)
            stats.failed_picks += 1
            continue

        # 3. Захват RGB кадра
        frame = camera.capture()   # (640, 640, 3) RGB uint8

        # 4. Детекция: class + bbox + yaw
        class_id_pred, x_w, y_w, yaw_deg = pipeline.run(frame)

        if class_id_pred < 0:
            print('  [ОШИБКА] Деталь не обнаружена детектором')
            conveyor.remove_part(part_handle)
            stats.failed_detections += 1
            continue

        print(f'  Детектор: {CLASS_NAMES[class_id_pred]}, '
              f'pos=({x_w:.3f}, {y_w:.3f}), yaw={yaw_deg:.1f}°')

        # 5. Оценка высоты из depth-карты
        bbox_px = pipeline.last_bbox_px
        if depth_sensor is not None:
            try:
                depth_frame = depth_sensor.capture_depth()
                # Глобальная оценка — весь кадр, без bbox.
                # Решает проблему несовпадения пикселей RGB/depth сенсоров.
                height_m = depth_est.estimate_height_global(depth_frame)
            except Exception as e:
                print(f'  [DEPTH ERROR] {e}')
                height_m = 0.03
        else:
            height_m = 0.03
        print(f'  Высота детали: {height_m*1000:.1f} мм')

        # 6. GraspPlanner → параметры захвата
        grasp_params = grasp_planner.resolve(
            class_id=class_id_pred,
            part_x=x_w,
            part_y=y_w,
            height_m=height_m,
            yaw_deg=yaw_deg,
            belt_z=BELT_Z,
        )
        print(f'  Пресет: "{grasp_params.preset_name}", '
              f'tilt={abs(grasp_params.euler[1]):.0f}°')

        # 7. Захват манипулятором
        grabbed = robot.pick_part(
            part_x=x_w,
            part_y=y_w,
            class_id=class_id_pred,
            grasp_params=grasp_params,
        )
        if not grabbed:
            print('  [ОШИБКА] Не удалось захватить деталь')
            #conveyor.remove_part(part_handle)
            stats.failed_picks += 1
            robot.go_home()
            continue

        # 8. Сортировка в бин
        robot.place_to_bin(class_id_pred)
        robot.go_home()

        # 9. Оценка корректности
        correct = (class_id_pred == class_id_gt)
        stats.record(correct)
        status = 'OK' if correct else 'НЕВЕРНО'
        print(f'  Результат: {status} '
              f'(GT={CLASS_NAMES[class_id_gt]}, '
              f'Pred={CLASS_NAMES[class_id_pred]})')

        # # 10. Пауза и удаление детали
        # for _ in range(100):
        #     sim.step()
        # conveyor.remove_part(part_handle)

    stats.print_summary()
    sim.stopSimulation()
    print('\nСимуляция завершена.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Запуск сортировки с детектором')
    parser.add_argument('n_parts',   type=int, nargs='?', default=20,
                        help='Количество деталей')
    parser.add_argument('--weights', type=str, default=None,
                        help='Путь к весам детектора')
    args = parser.parse_args()
    main(n_parts=args.n_parts, weights_path=args.weights)
