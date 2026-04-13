# Спецификация сцены CoppeliaSim

> Эталонный документ. Python-скрипты опираются на имена и параметры отсюда.
> Актуально для текущего состояния сцены (sorting_scene.ttt).

---

## Иерархия объектов (реальная сцена)

```
[Scene root]
│
├── DefaultCamera
├── XYZCameraProxy
├── DefaultLights
│
├── Floor
│   └── box                         ← пол (respondable, static)
│
├── IRB140                          ← манипулятор из Model Browser
│   ├── Script                      ← IK solver (Lua, не трогать)
│   ├── link1_visible
│   ├── manipulationSphereBase
│   │   └── manipulationSphere
│   │       └── target              ← IK target (Python двигает этот dummy)
│   ├── joint1 → link → joint2 → link → ... → joint6
│   │   └── link
│   │       └── connection
│   │           └── BaxterVacuumCupWhithGUI   ← захват (с GUI-скриптом)
│   │               ├── body
│   │               ├── sensor                ← proximity sensor захвата
│   │               ├── link                  ← force sensor
│   │               │   ├── loopClosureDummy2
│   │               │   └── tip               ← IK tip (TCP)
│   │               └── loopClosureDummy1     ← меняет parent при захвате
│
├── vision_sensor_main              ← Vision Sensor 640×640
│
├── conveyor                        ← generic conveyor belt (Model Browser)
│   ├── conveyor_sensor_arrival     ← proximity sensor (добавлен вручную)
│   └── conveyor_sensor_pickup      ← proximity sensor (добавлен вручную)
│
├── spawn_point                     ← Dummy, точка появления деталей
├── pickup_point                    ← Dummy, центр зоны захвата
│
├── bolt                            ← шаблон детали (static, hidden)
├── nut                             ← шаблон детали (static, hidden)
├── washer                          ← шаблон детали (static, hidden)
│
└── Bins
    ├── bin_bolt                    ← бин для болтов (0.2×0.2×0.2 м)
    ├── bin_nut                     ← бин для гаек
    └── bin_washer                  ← бин для шайб
```

---

## Пути объектов (для Python getObject)

| Python-переменная | Путь sim.getObject(...) | Примечание |
|---|---|---|
| `irb140` | `/IRB140` | корень манипулятора |
| `target_handle` | поиск по имени `target` в поддереве IRB140 | IK target |
| `tip_handle` | поиск по имени `tip` в поддереве IRB140 | TCP |
| `gripper_handle` | поиск `BaxterVacuumCupWhithGUI` в поддереве IRB140 | захват |
| `lcd1_handle` | поиск `loopClosureDummy1` в поддереве gripper | индикатор захвата |
| `vacuum_signal` | `sim.getObjectAlias(gripper, 4) + '_active'` | Int32 сигнал |
| `conveyor_handle` | `/conveyor` | корень конвейера |
| `arr_sensor_handle` | `/conveyor/conveyor_sensor_arrival` | |
| `pickup_sensor_handle` | `/conveyor/conveyor_sensor_pickup` | |
| `vision_handle` | `/vision_sensor_main` | |
| `spawn_handle` | `/spawn_point` | |
| `pickup_handle` | `/pickup_point` | |
| шаблоны | `/bolt`, `/nut`, `/washer` | источники для copyPaste |
| `bin_handles[0]` | `/Bins/bin_bolt` | |
| `bin_handles[1]` | `/Bins/bin_nut` | |
| `bin_handles[2]` | `/Bins/bin_washer` | |

> `target` и `tip` ищутся через `_find_in_subtree()` в `robot_control.py`,
> потому что лежат глубоко в иерархии на переменной глубине.

---

## IK система

Скрипт `/IRB140/Script` решает IK автоматически на каждом шаге симуляции:
- **Undamped** (pseudo-inverse, 10 итераций) → если не сошлось →
- **Damped** (DLS, damping=0.3, 99 итераций, allowError=true)

**Python не вызывает никаких IK-функций.** Достаточно:
```python
sim.setObjectPosition(target_handle, -1, [x, y, z])
sim.setObjectOrientation(target_handle, -1, [a, b, g])
```

---

## Вакуумный захват BaxterVacuumCupWhithGUI

Управление через **Int32 сигнал**:
```python
# Имя сигнала вычисляется автоматически в RobotController.__init__:
vacuum_alias = sim.getObjectAlias(gripper_handle, 4)
vacuum_signal = vacuum_alias + '_active'
# Пример: 'BaxterVacuumCupWhithGUI__41___active'

sim.setInt32Signal(vacuum_signal, 1)  # захват ВКЛ
sim.setInt32Signal(vacuum_signal, 0)  # захват ВЫКЛ
```

Логика Lua-скрипта захвата:
- Сигнал = 1: ищет ближайший respondable shape у `sensor`, при нахождении
  создаёт loop closure: `loopClosureDummy1` становится child найденной детали
- Сигнал = 0: разрывает loop closure, `loopClosureDummy1` возвращается к захвату

Проверка захвата из Python:
```python
parent = sim.getObjectParent(lcd1_handle)
grabbed = (parent != gripper_handle)  # True если деталь захвачена
```

---

## Конвейер

Управление скоростью:
```python
sim.setBufferProperty(
    conveyor_handle,
    'customData.__ctrl__',
    sim.packTable({'vel': 0.08})   # м/с, 0.0 = стоп
)
```

Чтение состояния:
```python
state = sim.readCustomTableData(conveyor_handle, '__state__')
```

---

## Создание деталей

Детали копируются из шаблонов сцены (`/bolt`, `/nut`, `/washer`):
```python
h = sim.copyPasteObjects([template_handle], 0)  # → список [new_handle]
sim.setObjectPosition(h[0], -1, spawn_pos)
sim.setObjectOrientation(h[0], -1, [0, 0, yaw])  # случайный yaw
sim.setObjectAlias(h[0], f'part_bolt_001')
```

Шаблоны должны быть в сцене как **static + respondable** объекты.
После сортировки копия удаляется: `sim.removeObject(part_handle)`.

---

## Vision Sensor

| Параметр | Значение |
|---|---|
| Имя | `vision_sensor_main` |
| Resolution | 640 × 640 |
| Projection | Perspective |
| FOV | 30° |
| Near / Far clip | 0.4 м / 0.7 м |
| **Explicit handling** | **ON** — рендер только по вызову из Python |
| Ориентация | смотрит вниз |

```python
sim.handleVisionSensor(vision_handle)
img, [w, h] = sim.getVisionSensorImg(vision_handle)
frame = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)
frame = np.flipud(frame)   # CoppeliaSim хранит снизу вверх
```

---

## Ключевые числа (уточнены по реальной сцене)

| Параметр | Значение |
|---|---|
| Высота поверхности ленты (`belt_surface_z`) | 0.131 м |
| Высота над деталью перед опусканием | `belt_surface_z + 0.15` |
| Высота захвата (`grasp_z`) | `belt_surface_z + 0.01` |
| `HOME_POS` | [0.0, −0.5, 0.65] |
| `ABOVE_PICKUP_POS` | [0.5, 0.0, 0.40] |
| `BIN_POSITIONS[0]` bolt | [−0.45, 0.20, 0.40] |
| `BIN_POSITIONS[1]` nut | [−0.45, −0.20, 0.40] |
| `BIN_POSITIONS[2]` washer | [−0.20, −0.45, 0.40] |
| `GRASP_EULER` | [0, 0, 0] (захват уже смотрит вниз по умолчанию) |

> Высота захвата зависит от класса детали. Регулируется через
> `GRASP_Z_OFFSET` в `RobotController`: bolt=0.035, nut=0.0, washer=0.0.
> `grasp_z = belt_surface_z + GRASP_Z_OFFSET[class_id]`
> Рабочие значения: bolt=0.06, nut=0.0, washer=0.0 (подобраны эмпирически).

---

## Поток одного цикла сортировки

```
1. conveyor.spawn_part(class_id)     → копия шаблона в spawn_point
2. conveyor.wait_for_part()          → конвейер везёт, сенсоры следят,
                                       замедление при arrival, стоп при pickup
3. camera.capture()                  → кадр 640×640 RGB numpy
4. detect_part(frame)                → class_id, x_world, y_world, yaw_deg
                                       (заглушка → реальный pipeline в Фазе 4)
5. robot.pick_part(x_w, y_w)        → над деталью → вакуум ON →
                                       опуститься → wait_for_grab → подняться
6. robot.place_to_bin(class_id)     → над бином → вакуум OFF → деталь падает
7. robot.go_home()
8. sim.removeObject(part_handle)    → деталь удаляется из сцены
```
