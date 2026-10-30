import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

random.seed(42)

SIZE = 128
SCALE = 3            # Draw larger, then resize for smoother lines
COUNT = 2000
OUTPUT = Path("dataset")

(OUTPUT / "images").mkdir(parents=True, exist_ok=True)

labels = []

def draw_stroke(draw, points, width):
    """Draw a multi-point pen stroke."""
    points = [(int(x * SCALE), int(y * SCALE))
              for x, y in points]

    if len(points) == 1:
        x, y = points[0]
        r = width * SCALE / 2
        draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
    else:
        draw.line(
            points,
            fill=255,
            width=width * SCALE,
            joint="curve"
        )



def random_stroke():
    """Generate a random line-like pen movement."""
    x = random.randint(5, SIZE - 5)
    y = random.randint(5, SIZE - 5)
    length = random.randint(5, 35)
    angle = random.uniform(-np.pi, np.pi)

    points = []
    for i in range(random.randint(3, 8)):
        t = i / (random.randint(3, 8) - 1) if False else i / 5
        px = x + np.cos(angle) * length * t
        py = y + np.sin(angle) * length * t
        px += random.uniform(-2, 2)
        py += random.uniform(-2, 2)
        points.append((px, py))

    return points

for i in range(COUNT):
    # Black background, white pen strokes.
    image = Image.new(
        "L", (SIZE * SCALE, SIZE * SCALE), 0
    )
    draw = ImageDraw.Draw(image)

    has_equals = random.random() < 0.5
    bbox = None

    # Draw distracting strokes first.
    for _ in range(random.randint(0, 7)):
        draw_stroke(
            draw,
            random_stroke(),
            random.randint(1, 3)
        )

    if has_equals:
        # Leave enough space for both lines.
        cx = random.randint(25, 103)
        cy = random.randint(20, 108)

        half_width = random.randint(8, 22)
        gap = random.randint(6, 13)
        slant = random.uniform(-2, 2)
        thickness = random.randint(1, 3)

        x1 = cx - half_width
        x2 = cx + half_width

        # Upper line.
        upper = [
            (x1, cy - gap / 2 - slant),
            (cx, cy - gap / 2),
            (x2, cy - gap / 2 + slant)
        ]

        # Lower line.
        lower = [
            (x1, cy + gap / 2 - slant),
            (cx, cy + gap / 2),
            (x2, cy + gap / 2 + slant)
        ]

        draw_stroke(draw, upper, thickness)
        draw_stroke(draw, lower, thickness)

        # Approximate bounding box around the sign.
        left = x1 - thickness / 2
        right = x2 + thickness / 2
        top = cy - gap / 2 - abs(slant) - thickness / 2
        bottom = cy + gap / 2 + abs(slant) + thickness / 2

        bbox = [
            (left + right) / 2 / SIZE,
            (top + bottom) / 2 / SIZE,
            (right - left) / SIZE,
            (bottom - top) /SIZE
        ]

     # Downsample for anti-aliased strokes.
    image = image.resize(
        (SIZE, SIZE), Image.Resampling.LANCZOS
    )

    filename = f"{i:05d}.png"
    image.save(OUTPUT / "images" / filename)

    labels.append({
        "image": f"images/{filename}",
        "present": int(has_equals),
        "bbox": bbox
    })

with open(OUTPUT / "labels.json", "w") as f:
    json.dump(labels, f, indent=2)

print(f"Generated {COUNT} images.")
print(f"Positive examples: {sum(x['present'] for x in labels)}")
print(f"Labels saved to {OUTPUT / 'labels.json'}")
