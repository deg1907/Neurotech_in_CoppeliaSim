# check_annotations.py — визуализация bbox поверх изображений
import cv2
import os

img_dir = r'C:\Users\deg19\Desktop\dissertation_w_Claude\dataset\images\raw'
lbl_dir = r'C:\Users\deg19\Desktop\dissertation_w_Claude\dataset\labels\raw'

for stem in ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009']:
    img = cv2.imread(os.path.join(img_dir, f'{stem}.png'))
    h, w = img.shape[:2]

    with open(os.path.join(lbl_dir, f'{stem}.txt')) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'cls={int(cls)}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    cv2.imwrite(f'check_{stem}.png', img)
    print(f'Saved check_{stem}.png')