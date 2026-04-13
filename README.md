# Система технического зрения для автоматизированной сортировки деталей в CoppeliaSim

Магистерская диссертация — НГТУ, факультет мехатроники и автоматизации.

Разработка системы технического зрения на основе глубокого обучения для автоматизированной сортировки деталей роботом-манипулятором в виртуальной среде CoppeliaSim.

---

## Описание

Система выполняет полный цикл сортировки промышленных деталей:

1. **Захват кадра** — RGB и Depth Vision Sensor 640×640 над конвейером
2. **Детекция** — кастомный CNN определяет класс детали и bounding box
3. **Оценка ориентации** — yaw-регрессор CNN определяет угол поворота детали
4. **Оценка высоты** — DepthEstimator находит высоту верхней грани из depth-карты
5. **Планирование захвата** — GraspPlanner читает `grasp_presets.yaml` и формирует параметры захвата
6. **Перекладывание** — манипулятор ABB IRB 140 захватывает деталь и кладёт в нужный бин

### Классы деталей

| Класс | Описание |
|---|---|
| `gaika` | Гайка шестигранная с буртиком ГОСТ 8918-69 |
| `vilka` | Вилка с резьбовым отверстием ГОСТ 12470-67 |
| `vtulka` | Втулка (цилиндр с фланцем) |

---

## Архитектура

```
RGB-кадр 640×640          Depth-кадр 640×640
    ↓                              ↓
Детектор (CNN)           DepthEstimator
  bbox + class             → height_m
    ↓
Yaw-регрессор (CNN)
  crop 64×64 → yaw (°)
    ↓
GraspPlanner.resolve(class, height_m, yaw)
  → grasp_presets.yaml → GraspParams(pos, euler)
    ↓
pixel → world → IK → захват с tilt → бин
```

---

## Стек технологий

| Компонент | Технология |
|---|---|
| Симулятор | CoppeliaSim v4.10, Python ZMQ Remote API |
| Манипулятор | ABB IRB 140 (6-DOF), встроенный IK solver |
| Захват | Вакуумный (proximity sensor + sim.setObjectParent) |
| CV / ML | PyTorch, кастомные CNN (без pretrained) |
| Датасет | 3000–5000 синтетических изображений с domain randomization |
| Формат аннотаций | YOLO txt + отдельные yaw-метки |

---

## Структура проекта

```
├── coppeliasim/
│   ├── scene/              # Сцена CoppeliaSim (.ttt)
│   ├── scripts/            # Управление роботом, конвейером, захватом
│   └── models/             # STL-модели деталей
├── dataset/
│   ├── generation/         # Генерация датасета + domain randomization + автоаннотация
│   ├── images/             # train / val / test
│   ├── labels/             # YOLO-формат
│   └── yaw_labels/         # yaw-аннотации
├── models/
│   ├── detector/           # Архитектура, loss, обучение, метрики, инференс
│   ├── yaw_regressor/      # То же для yaw-регрессора
│   └── weights/            # Сохранённые веса .pt
├── integration/
│   ├── pipeline.py         # Полный pipeline: кадр → захват
│   ├── calibration.py      # Калибровка камера→робот (pixel→world)
│   └── sorting_controller.py
├── experiments/
│   ├── ablation_study.py   # Ablation study по domain randomization
│   └── compare_models.py
├── dissertation/           # Текст диссертации, рисунки, список литературы
├── requirements.txt
└── README.md
```

---

## Установка и запуск

### Зависимости

```bash
pip install -r requirements.txt
```

Основные пакеты: `torch`, `torchvision`, `numpy`, `opencv-python`, `PyYAML`, `coppeliasim-zmqremoteapi-client`.

### Генерация датасета

```bash
# Запустить CoppeliaSim со сценой sorting_scene.ttt, затем:
python dataset/generation/generate_dataset.py
```

### Обучение детектора

```bash
python models/detector/train.py
```

### Обучение yaw-регрессора

```bash
python models/yaw_regressor/train.py
```

### Запуск полного pipeline

```bash
# CoppeliaSim должен быть запущен со сценой
python coppeliasim/scripts/main_loop.py
```

### Обучение в Google Colab

Открой `colab_train_detector.ipynb` — всё необходимое уже настроено.

---

## Метрики

| Задача | Метрика |
|---|---|
| Детекция | mAP@0.5, mAP@0.5:0.95, Precision, Recall |
| Yaw-регрессия | MAE (градусы) |
| Система в целом | Accuracy сортировки (%), throughput (дет/мин) |

---

## Ablation Study

Исследование влияния domain randomization на качество детекции:

| Эксперимент | DR | Ожидаемый эффект |
|---|---|---|
| Baseline | Без DR | Нижняя граница метрик |
| Partial DR | Только свет | Промежуточный результат |
| Full DR | Свет + текстуры + шум | Лучшая генерализация |

---

## Научная новизна

1. Кастомная CNN-архитектура детектора, оптимизированная для синтетических промышленных данных
2. Замкнутый цикл генерации датасета в CoppeliaSim с автоаннотацией через 3D→2D проекцию вершин
3. Количественный ablation study влияния domain randomization на метрики детекции и accuracy сортировки
4. Двухступенчатый pipeline «детекция + yaw-регрессия + depth» для планирования захвата деталей в произвольной позе
5. Конфигурируемый планировщик захвата через `grasp_presets.yaml` — смена деталей без изменения кода
