# CoppeliaSim ZMQ Remote API — справочник используемых функций

Проверено на CoppeliaSim v4.10, Python ZMQ Remote API.

---

## Объект и позиция

```python
sim.getObject('/name')                        # handle по пути
sim.getObjectPosition(handle, -1)            # [x, y, z] мировая СК
sim.setObjectPosition(handle, -1, [x,y,z])
sim.getObjectOrientation(handle, -1)         # [alpha, beta, gamma] Эйлер
sim.setObjectOrientation(handle, -1, [a,b,g])
sim.getObjectMatrix(handle, -1)              # список 12 float → reshape(3,4) = [R|t]
sim.setObjectMatrix(handle, -1, matrix_12)
sim.getObjectParent(handle)                  # handle родителя
sim.setObjectParent(handle, parent, keepPos) # перепривязать объект
sim.getObjectAlias(handle, 0)                # имя объекта (без суффикса)
sim.getObjectAlias(handle, 4)               # полное имя с суффиксом __NNN__
sim.setObjectAlias(handle, 'name')
sim.getObjectsInTree(root, sim.handle_all, 0)  # все объекты в поддереве
sim.getObjectType(handle)                    # sim.object_shape_type и др.
```

## Bounding Box объекта (AABB в локальной СК)

```python
# Через getObjectFloatParam — AABB в ЛОКАЛЬНОЙ СК объекта:
min_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
min_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
min_z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
max_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x)
max_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y)
max_z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z)
# Затем трансформировать 8 углов через getObjectMatrix → мировые координаты

# Альтернатива — модельный bbox (если объект — модель):
# sim.objfloatparam_modelbbox_min/max_x/y/z
```

**НЕ существует:** `sim.getObjectBoundingBox()`, `sim.getShapeBoundingBox()`

## Vision Sensor

```python
sim.handleVisionSensor(handle)               # триггер рендера
sim.getVisionSensorImg(handle)               # (bytes, [w, h]) — RGB uint8
sim.getVisionSensorDepth(handle)             # (list[float], [w, h]) — нормированные [0,1]
# depth_m = near + (far - near) * depth_norm

sim.getObjectFloatParam(handle, sim.visionfloatparam_perspective_angle)  # FOV радиан
sim.getObjectFloatParam(handle, sim.visionfloatparam_near_clipping)
sim.getObjectFloatParam(handle, sim.visionfloatparam_far_clipping)
sim.getObjectIntParam(handle, sim.visionintparam_resolution_x)
sim.getObjectIntParam(handle, sim.visionintparam_resolution_y)
```

## Копирование / удаление объектов

```python
handles = sim.copyPasteObjects([template_handle], 0)  # returns list
sim.removeObject(handle)
```

## Цвет формы

```python
sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, [r, g, b])
sim.getObjectColor(handle, 0, sim.colorcomponent_ambient_diffuse)  # → [r, g, b]
```

## Proximity sensor

```python
result, dist, detected_handle, *_ = sim.readProximitySensor(sensor_handle)
# result == 1 если что-то обнаружено
```

## Сигналы (вакуумный захват)

```python
sim.setInt32Signal('signal_name', 1)   # активировать
sim.setInt32Signal('signal_name', 0)   # деактивировать
```

## Управление симуляцией

```python
client.setStepping(True)     # синхронный режим
sim.startSimulation()
sim.step()                   # один шаг
sim.stopSimulation()
sim.getSimulationTime()      # текущее время симуляции (с)
```

## Конвейер (setBufferProperty)

```python
sim.setBufferProperty(
    conveyor_handle,
    'customData.__ctrl__',
    sim.packTable({'vel': 0.08})
)
```

## IK / манипулятор

```python
sim.setObjectPosition(target_handle, -1, [x, y, z])
sim.setObjectOrientation(target_handle, -1, [a, b, g])
# IK решается автоматически скриптом сцены на каждом sim.step()
```

## Матрица позы — формат

```python
m = sim.getObjectMatrix(handle, -1)  # список 12 float
M = np.array(m).reshape(3, 4)
# M = [R | t], где R — 3x3 ротация (столбцы = оси объекта в мировой СК)
#              t — позиция объекта в мировой СК
# Преобразование точки из локальной в мировую: v_world = M @ [v_local, 1]
# Обратная (мир→объект): Rt = R.T; t_inv = -Rt @ t
```
