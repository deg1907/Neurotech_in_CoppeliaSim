"""
Вычисление центра масс и матрицы инерции STL-деталей.

Зависимости:
    pip install trimesh numpy

Использование:
    python calc_inertia.py                  # все .stl в папке models/
    python calc_inertia.py gaika.stl        # конкретный файл
    python calc_inertia.py --density 7800   # своя плотность (кг/м^3)
    python calc_inertia.py --units mm       # если STL в миллиметрах
"""

import argparse
import io
import sys
from pathlib import Path

# Принудительно UTF-8 для stdout на Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import trimesh

# Плотность стали по умолчанию, кг/м^3
DEFAULT_DENSITY = 7800.0


def compute_mesh_properties(stl_path: str, density: float, scale: float) -> dict:
    """
    Вычисляет массу, центр масс и матрицу инерции для STL-сетки.

    Использует trimesh: загружает замкнутую полигональную сетку,
    задаёт плотность и получает точные значения через интегрирование
    по тетраэдральному разбиению (метод Миртича 1996).

    Args:
        stl_path: путь к .stl файлу
        density:  плотность материала, кг/м^3
        scale:    суммарный масштабный коэффициент
                  (units_scale * scene_scale, например 1e-3 * 2 для мм + x2 в сцене)

    Returns:
        Словарь: mass, com (3,), inertia (3x3), volume
    """
    part = trimesh.load_mesh(stl_path)

    # Применяем масштаб (единицы + масштаб сцены)
    if scale != 1.0:
        part.apply_scale(scale)

    # Проверка замкнутости сетки (важно для корректного расчёта)
    if not part.is_watertight:
        print(f"  [!] Предупреждение: сетка не замкнута — результаты могут быть неточными")

    # Задаём плотность и получаем инерционные свойства
    part.density = density

    volume = part.volume                    # м^3
    mass   = part.mass                      # кг = density * volume
    com    = part.center_mass               # (3,) м
    inertia = part.moment_inertia           # (3,3) кг·м^2, относительно COM

    return {
        "mass":    mass,
        "volume":  volume,
        "com":     com,
        "inertia": inertia,
    }


def print_report(name: str, props: dict) -> None:
    """Выводит отчёт по детали в читаемом формате."""
    m = props["mass"]
    v = props["volume"]
    c = props["com"]
    I = props["inertia"]

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Деталь : {name}")
    print(f"{sep}")
    print(f"  Объём  : {v * 1e6:.4f} см^3   ({v:.6e} м^3)")
    print(f"  Масса  : {m * 1e3:.4f} г     ({m:.6e} кг)")
    print(f"  Центр масс (м):")
    print(f"    x = {c[0]:.6e}")
    print(f"    y = {c[1]:.6e}")
    print(f"    z = {c[2]:.6e}")
    print(f"  Матрица инерции (кг·м^2) относительно COM:")
    for row in range(3):
        row_str = "    " + "  ".join(f"{I[row, col]:+.6e}" for col in range(3))
        print(row_str)
    print(f"\n  Диагональные моменты инерции:")
    print(f"    Ixx = {I[0,0]:.6e} кг·м^2")
    print(f"    Iyy = {I[1,1]:.6e} кг·м^2")
    print(f"    Izz = {I[2,2]:.6e} кг·м^2")

    # CoppeliaSim: Body > Inertia matrix — задаётся как [Ixx, Iyy, Izz] + внедиагональные
    print(f"\n  Для CoppeliaSim (Body > Inertia):")
    print(f"    Ixx={I[0,0]:.6e}  Iyy={I[1,1]:.6e}  Izz={I[2,2]:.6e}")
    print(f"    Ixy={I[0,1]:.6e}  Ixz={I[0,2]:.6e}  Iyz={I[1,2]:.6e}")
    print(f"    COM (м): [{c[0]:.6e}, {c[1]:.6e}, {c[2]:.6e}]")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Вычисление инерции и COM для STL-деталей"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="STL-файлы (если не указаны — все *.stl в текущей папке)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_DENSITY,
        help=f"Плотность материала кг/м^3 (по умолч. {DEFAULT_DENSITY} — сталь)",
    )
    parser.add_argument(
        "--units",
        choices=["m", "mm"],
        default="m",
        help="Единицы STL-файла: m — метры (по умолч.), mm — миллиметры",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help="Масштаб модели в сцене (напр. 2.0 если модель увеличена в 2 раза)",
    )
    args = parser.parse_args()

    units_scale = 1e-3 if args.units == "mm" else 1.0
    scale = units_scale * args.scene_scale

    if args.files:
        stl_files = [Path(f) for f in args.files]
    else:
        stl_files = sorted(Path(__file__).parent.glob("*.stl"))

    if not stl_files:
        print("STL-файлы не найдены. Положи .stl рядом со скриптом или укажи пути.")
        sys.exit(1)

    print(f"\nПлотность: {args.density} кг/м^3")
    print(f"Единицы STL: {args.units}  |  Масштаб сцены: x{args.scene_scale}")

    for stl_path in stl_files:
        if not stl_path.exists():
            print(f"\n[!] Файл не найден: {stl_path}")
            continue
        props = compute_mesh_properties(str(stl_path), args.density, scale)
        print_report(stl_path.name, props)


if __name__ == "__main__":
    main()
