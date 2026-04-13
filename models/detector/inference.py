"""
Инференс детектора на одном изображении.

Запуск:
  python inference.py --weights ../../weights/detector_best.pt --image path/to/img.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from architecture import Detector

# Названия классов и цвета bbox (BGR)
CLASS_NAMES: dict[int, str] = {0: 'gaika', 1: 'vilka', 2: 'vtulka'}
CLASS_COLORS: dict[int, tuple] = {
    0: (60,  200, 60),    # gaika — зелёный
    1: (220, 60,  60),    # vilka — красный
    2: (60,  60,  220),   # vtulka — синий
}


def load_model(
    weights_path: str,
    num_classes: int = 3,
    device: str = 'cpu',
) -> Detector:
    """
    Загрузить модель из checkpoint файла.

    Args:
        weights_path: путь к .pt файлу (сохранён в train.py)
        num_classes:  число классов
        device:       'cpu' или 'cuda'

    Returns:
        Detector в режиме eval
    """
    dev = torch.device(device)
    model = Detector(num_classes=num_classes).to(dev)
    ckpt = torch.load(weights_path, map_location=dev)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Модель загружена: {weights_path} (эпоха {ckpt.get("epoch", "?")})')
    return model


def preprocess(image_bgr: np.ndarray, img_size: int = 640) -> torch.Tensor:
    """
    Подготовить изображение к инференсу.

    Args:
        image_bgr: numpy array (H, W, 3), BGR
        img_size:  целевой размер стороны

    Returns:
        тензор (1, 3, img_size, img_size), float32, [0,1]
    """
    img = cv2.resize(image_bgr, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def postprocess(
    result: dict | None,
    orig_h: int,
    orig_w: int,
) -> dict | None:
    """
    Перевести нормализованный bbox в пиксельные координаты исходного изображения.

    Args:
        result: вывод model.predict() — {'class', 'conf', 'bbox': [x_c,y_c,w,h]}
        orig_h: высота исходного изображения (пиксели)
        orig_w: ширина исходного изображения (пиксели)

    Returns:
        dict {'class', 'conf', 'bbox_px': [x1, y1, x2, y2]} или None
    """
    if result is None:
        return None

    x_c, y_c, w, h = result['bbox']
    x1 = int((x_c - w / 2) * orig_w)
    y1 = int((y_c - h / 2) * orig_h)
    x2 = int((x_c + w / 2) * orig_w)
    y2 = int((y_c + h / 2) * orig_h)

    # Зажимаем в границы изображения
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(orig_w, x2);  y2 = min(orig_h, y2)

    return {
        'class':   result['class'],
        'conf':    result['conf'],
        'bbox_px': [x1, y1, x2, y2],
    }


def detect(
    model: Detector,
    image_bgr: np.ndarray,
    conf_thresh: float = 0.5,
    device: str = 'cpu',
) -> dict | None:
    """
    Полный pipeline: изображение → предсказание.

    Args:
        model:      загруженный Detector
        image_bgr:  numpy array (H, W, 3), BGR
        conf_thresh: порог объектности
        device:     устройство модели

    Returns:
        {'class': int, 'conf': float, 'bbox_px': [x1, y1, x2, y2]}
        или None если объект не найден
    """
    orig_h, orig_w = image_bgr.shape[:2]
    tensor = preprocess(image_bgr).to(torch.device(device))
    results = model.predict(tensor, conf_thresh=conf_thresh)
    return postprocess(results[0], orig_h, orig_w)


def draw_detection(
    image_bgr: np.ndarray,
    result: dict,
    class_names: dict | None = None,
) -> np.ndarray:
    """
    Нарисовать bbox и подпись на изображении.

    Args:
        image_bgr:   оригинальное изображение (копируется внутри)
        result:      вывод detect() с полем 'bbox_px'
        class_names: {0: 'bolt', 1: 'nut', 2: 'washer'} (по умолчанию CLASS_NAMES)

    Returns:
        копия изображения с нанесёнными аннотациями
    """
    if class_names is None:
        class_names = CLASS_NAMES

    img = image_bgr.copy()
    cls_id  = result['class']
    conf    = result['conf']
    x1, y1, x2, y2 = result['bbox_px']
    color   = CLASS_COLORS.get(cls_id, (200, 200, 200))
    label   = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

    # Подложка под текст
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Инференс детектора')
    parser.add_argument('--weights', type=str, required=True,
                        help='Путь к файлу весов .pt')
    parser.add_argument('--image',   type=str, required=True,
                        help='Путь к изображению')
    parser.add_argument('--conf',    type=float, default=0.5,
                        help='Порог объектности')
    parser.add_argument('--device',  type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out',     type=str, default=None,
                        help='Путь для сохранения результата (опционально)')
    args = parser.parse_args()

    model     = load_model(args.weights, num_classes=3, device=args.device)
    image_bgr = cv2.imread(args.image)

    if image_bgr is None:
        print(f'Ошибка: не удалось открыть {args.image}')
        sys.exit(1)

    result = detect(model, image_bgr, conf_thresh=args.conf, device=args.device)

    if result is None:
        print('Объект не обнаружен')
    else:
        cls_name = CLASS_NAMES.get(result['class'], str(result['class']))
        print(f'Класс: {cls_name}  conf={result["conf"]:.3f}  '
              f'bbox={result["bbox_px"]}')

        annotated = draw_detection(image_bgr, result)
        out_path  = args.out or str(Path(args.image).stem) + '_result.png'
        cv2.imwrite(out_path, annotated)
        print(f'Результат сохранён: {out_path}')
