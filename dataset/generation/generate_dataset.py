"""
Генерация синтетического датасета через CoppeliaSim.

Алгоритм одного кадра:
  1. Выбрать случайный класс детали
  2. Скопировать шаблон в pickup_point с:
       - случайным yaw [0, 360°)
       - случайным XY-смещением в пределах конвейера
  3. Прогнать несколько шагов симуляции (деталь оседает на ленте)
  4. Применить domain randomization (освещение, цвет)
  5. Захватить кадр с Vision Sensor
  6. Применить image-level DR (шум)
  7. Вычислить аннотацию через Annotator (YOLO bbox + yaw)
  8. Сохранить изображение и аннотации
  9. Удалить деталь

Запуск:
  python generate_dataset.py --n 3000 --dr full --split
  python generate_dataset.py --n 1000 --dr none          # baseline
  python generate_dataset.py --n 1000 --dr light         # partial DR

Аргументы:
  --n      N     количество изображений (default: 3000)
  --dr     MODE  none | light | full (default: full)
  --split        разбить на train/val/test (70/20/10) после генерации
  --out    DIR   корневая директория датасета (default: ../../dataset)
"""

import sys
import os
import argparse
import random
import shutil
import math
from pathlib import Path

import numpy as np
import cv2  # для сохранения изображений

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'coppeliasim' / 'scripts'))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from annotator import Annotator
from domain_random import DRMode, LightRandomizer, randomize_part_color, apply_image_dr
from conveyor import ConveyorController

# ------------------------------------------------------------------ #
#  Константы                                                          #
# ------------------------------------------------------------------ #

CLASS_NAMES = {0: 'gaika', 1: 'vilka', 2: 'vtulka'}

# Шаблоны деталей в сцене
TEMPLATE_PATHS = {0: '/gaika', 1: '/vilka', 2: '/vtulka'}

# Случайное XY-смещение относительно pickup_point (метры)
# Детали появляются в разных местах конвейерной зоны
SPAWN_X_RANGE: tuple[float, float] = (-0.07, 0.07)   # ±70 мм по ширине ленты
SPAWN_Y_RANGE: tuple[float, float] = (-0.05, 0.05)   # ±50 мм по длине

# Шагов симуляции для оседания детали на ленте.
# 80 шагов — достаточно для стабилизации при любой начальной ориентации.
SETTLE_STEPS: int = 80

# Высота спавна над поверхностью ленты (деталь падает вниз)
SPAWN_Z_OFFSET: float = 0.05   # 50 мм выше ленты


# ------------------------------------------------------------------ #
#  Вспомогательные функции                                            #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Генерация датасета')
    p.add_argument('--n',     type=int,   default=3000,      help='Число изображений')
    p.add_argument('--dr',    type=str,   default='full',     help='none | light | full')
    p.add_argument('--split', action='store_true',            help='Разбить на train/val/test')
    p.add_argument('--out',   type=str,   default=None,       help='Корень датасета')
    return p.parse_args()


def get_dr_mode(mode_str: str) -> DRMode:
    return {'none': DRMode.NONE, 'light': DRMode.LIGHT, 'full': DRMode.FULL}.get(
        mode_str.lower(), DRMode.FULL
    )


def make_dirs(dataset_root: Path) -> tuple[Path, Path, Path]:
    """Создать директории для изображений и меток."""
    img_dir  = dataset_root / 'images' / 'raw'
    lbl_dir  = dataset_root / 'labels' / 'raw'
    yaw_dir  = dataset_root / 'yaw_labels' / 'raw'
    for d in (img_dir, lbl_dir, yaw_dir):
        d.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir, yaw_dir


def split_dataset(dataset_root: Path, ratios: tuple = (0.7, 0.2, 0.1)) -> None:
    """
    Разбить датасет на train/val/test.

    Перемещает файлы из images/raw → images/train|val|test
    и аналогично для labels/ и yaw_labels/.
    """
    splits = ['train', 'val', 'test']
    img_raw = dataset_root / 'images' / 'raw'
    all_imgs = sorted(img_raw.glob('*.png'))
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    boundaries = [0, n_train, n_train + n_val, n]

    for split_idx, split_name in enumerate(splits):
        subset = all_imgs[boundaries[split_idx]:boundaries[split_idx + 1]]
        for src_dir, dst_base in [
            ('images', 'images'),
            ('labels', 'labels'),
            ('yaw_labels', 'yaw_labels'),
        ]:
            dst_dir = dataset_root / dst_base / split_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path in subset:
                stem = img_path.stem
                if src_dir == 'images':
                    src = dataset_root / src_dir / 'raw' / f'{stem}.png'
                    dst = dst_dir / f'{stem}.png'
                else:
                    src = dataset_root / src_dir / 'raw' / f'{stem}.txt'
                    dst = dst_dir / f'{stem}.txt'
                if src.exists():
                    shutil.copy2(src, dst)

    print(f'Split: train={n_train}, val={n_val}, test={n - n_train - n_val}')


# ------------------------------------------------------------------ #
#  Спавн детали                                                       #
# ------------------------------------------------------------------ #

def spawn_part(sim, class_id: int, pickup_pos: list[float]) -> int:
    """
    Скопировать шаблон детали и разместить в зоне захвата.

    Args:
        sim:        Remote API
        class_id:   0=bolt, 1=nut, 2=washer
        pickup_pos: [x, y, z] центра зоны захвата (мировая СК)

    Returns:
        handle созданной детали
    """
    template = sim.getObject(TEMPLATE_PATHS[class_id])
    h_list = sim.copyPasteObjects([template], 0)
    part_handle = h_list[0]

    # Случайное смещение по XY и случайная 3D-ориентация.
    # Деталь спавнится выше ленты и падает вниз, оседая на произвольную грань.
    dx   = random.uniform(*SPAWN_X_RANGE)
    dy   = random.uniform(*SPAWN_Y_RANGE)
    roll  = random.uniform(-math.pi, math.pi)
    pitch = random.uniform(-math.pi, math.pi)
    yaw   = random.uniform(0, 2 * math.pi)

    spawn_pos = [
        pickup_pos[0] + dx,
        pickup_pos[1] + dy,
        pickup_pos[2] + SPAWN_Z_OFFSET,
    ]
    sim.setObjectPosition(part_handle, -1, spawn_pos)
    sim.setObjectOrientation(part_handle, -1, [roll, pitch, yaw])

    return part_handle


# ------------------------------------------------------------------ #
#  Главный цикл генерации                                             #
# ------------------------------------------------------------------ #

def generate(n: int, dr_mode: DRMode, dataset_root: Path) -> None:
    """
    Генерировать n изображений датасета.

    Args:
        n:            количество изображений
        dr_mode:      режим domain randomization
        dataset_root: корень датасета (содержит images/, labels/, yaw_labels/)
    """
    img_dir, lbl_dir, yaw_dir = make_dirs(dataset_root)

    # Подключение к CoppeliaSim
    client = RemoteAPIClient()
    sim    = client.require('sim')
    client.setStepping(True)
    sim.startSimulation()
    conveyor = ConveyorController(sim)
    conveyor.stop()  # Остановить конвейер на всякий случай, если он был запущен вручну
    # Handles объектов сцены
    vision_handle  = sim.getObject('/vision_sensor_main')
    pickup_handle  = sim.getObject('/pickup_point')

    pickup_pos = sim.getObjectPosition(pickup_handle, -1)

    # Аннотатор и рандомизатор освещения
    annotator      = Annotator(sim, vision_handle)
    light_rand     = LightRandomizer(sim) if dr_mode != DRMode.NONE else None

    # Балансировка классов: поровну каждого
    class_ids = [i % 3 for i in range(n)]
    random.shuffle(class_ids)

    saved = 0
    skipped = 0

    print(f'Генерация {n} изображений | DR: {dr_mode.name}')

    for idx, class_id in enumerate(class_ids):
        # --- Domain randomization освещения ---
        if light_rand and dr_mode in (DRMode.LIGHT, DRMode.FULL):
            light_rand.randomize()

        # --- Фоновый кадр (пустая лента, ДО спавна детали) ---
        # Снимается ПОСЛЕ рандомизации освещения — чтобы diff был чистым
        sim.handleVisionSensor(vision_handle)
        img_bg, [w_bg, h_bg] = sim.getVisionSensorImg(vision_handle)
        background = np.frombuffer(img_bg, dtype=np.uint8).reshape(h_bg, w_bg, 3)
        background = np.flipud(background).copy()

        # --- Спавн детали ---
        part_handle = spawn_part(sim, class_id, pickup_pos)

        # --- Domain randomization цвета детали ---
        if dr_mode == DRMode.FULL:
            randomize_part_color(sim, part_handle, class_id)

        # --- Оседание ---
        for _ in range(SETTLE_STEPS):
            sim.step()

        # --- Захват кадра ---
        sim.handleVisionSensor(vision_handle)
        img_bytes, [w, h] = sim.getVisionSensorImg(vision_handle)
        frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)
        frame = np.flipud(frame).copy()  # исправить ориентацию

        # --- Аннотация через background subtraction ---
        # bbox_from_diff даёт точный пиксельный bbox при любой ориентации детали,
        # полностью минуя AABB API CoppeliaSim (objbbox, modelbbox, getObjectsInTree)
        bbox_px = Annotator.bbox_from_diff(frame, background)
        if bbox_px is None:
            # Деталь вышла за пределы кадра или слилась с фоном
            sim.removeObject(part_handle)
            skipped += 1
            continue

        ann = annotator.annotate_from_bbox(bbox_px, part_handle, class_id)
        if ann is None:
            sim.removeObject(part_handle)
            skipped += 1
            continue

        # --- Image-level DR (шум) ---
        frame_dr = apply_image_dr(frame, dr_mode)

        # --- Сохранение ---
        stem = f'{idx:05d}'

        # Изображение (BGR для OpenCV)
        img_bgr = frame_dr[:, :, ::-1]
        cv2.imwrite(str(img_dir / f'{stem}.png'), img_bgr)

        # YOLO-аннотация
        Annotator.save_yolo_label(
            str(lbl_dir / f'{stem}.txt'),
            [ann['yolo']]
        )

        # Yaw-аннотация
        Annotator.save_yaw_label(
            str(yaw_dir / f'{stem}.txt'),
            class_id,
            ann['yaw']
        )

        # --- Сброс освещения ---
        if light_rand:
            light_rand.reset()

        # --- Удалить деталь ---
        sim.removeObject(part_handle)
        saved += 1

        if (saved) % 10 == 0 or saved == 1:
            print(f'  [{saved}/{n}] сохранено, пропущено: {skipped}')

    sim.stopSimulation()
    print(f'\nГотово: {saved} изображений, {skipped} пропущено')
    print(f'Датасет: {dataset_root}')


# ------------------------------------------------------------------ #
#  Точка входа                                                        #
# ------------------------------------------------------------------ #

def main() -> None:

    args = parse_args()
    dr_mode = get_dr_mode(args.dr)

    # Корень датасета: по умолчанию ../../dataset относительно этого файла
    if args.out:
        dataset_root = Path(args.out)
    else:
        dataset_root = Path(__file__).parent.parent.parent / 'dataset'

    generate(args.n, dr_mode, dataset_root)

    if args.split:
        print('\nРазбивка на train/val/test...')
        split_dataset(dataset_root)


if __name__ == '__main__':
    main()
