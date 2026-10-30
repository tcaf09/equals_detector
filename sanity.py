import cv2, random
from pathlib import Path

imgs = random.sample(list(Path("dataset/images/train").glob("*.png")), 8)
for p in imgs:
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    lbl = Path("dataset/labels/train")/(p.stem + ".txt")
    for line in lbl.read_text().splitlines():
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
        x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(f"check_{p.name}", img)
