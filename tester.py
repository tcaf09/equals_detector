from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")
out = Path("test_out"); out.mkdir(exist_ok=True)

for p in list(Path("dataset/images/val").glob("*.png"))[5:]:
    img = Image.open(p).convert("RGB")
    result = model.predict(img, imgsz=320, conf=0.4, verbose=False)[0]
    draw = ImageDraw.Draw(img)
    for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{conf:.2f}", fill="red")
    img.save(out / p.name)
    print(p.name, f"{len(result.boxes)} detection(s)")
