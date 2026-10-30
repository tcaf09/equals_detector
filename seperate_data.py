import json
import math
import shutil
from pathlib import Path

negatives = []
positives = []

with open("dataset/labels.json", "r", encoding="utf-8") as labels:
    data = json.load(labels)
    for label in data:
        if label["present"] == 0:
            negatives.append(label)
        else:
            positives.append(label)

for split in ("train", "val"):
    Path("dataset/images", split).mkdir(parents=True, exist_ok=True)
    Path("dataset/labels", split).mkdir(parents=True, exist_ok=True)

negTrain = math.floor(len(negatives) * (80 / 100))
posTrain = math.floor(len(positives) * (80 / 100))

for label in negatives[:negTrain]:
    shutil.copy("dataset/" + label["image"], "dataset/images/train/")
    labelName = "dataset/labels/train/" + Path(label["image"]).stem + ".txt"
    with open(labelName, "w", encoding="utf-8") as f:
        f.write("")

for label in negatives[negTrain:]:
    shutil.copy("dataset/" + label["image"], "dataset/images/val/")
    labelName = "dataset/labels/val/" + Path(label["image"]).stem + ".txt"
    with open(labelName, "w", encoding="utf-8") as f:
        f.write("")

for label in positives[:posTrain]:
    shutil.copy("dataset/" + label["image"], "dataset/images/train/")
    labelName = "dataset/labels/train/" + Path(label["image"]).stem + ".txt"
    with open(labelName, "w", encoding="utf-8") as f:
        f.write("0 " + " ".join(map(str, label["bbox"])) + "\n")

for label in positives[posTrain:]:
    shutil.copy("dataset/" + label["image"], "dataset/images/val/")
    labelName = "dataset/labels/val/" + Path(label["image"]).stem + ".txt"
    with open(labelName, "w", encoding="utf-8") as f:
        f.write("0 " + " ".join(map(str, label["bbox"])) + "\n")

print(f"Train: {len(negatives[:negTrain]) + len(positives[:posTrain])} images")
print(f"Val: {len(negatives[negTrain:]) + len(positives[posTrain:])} images")
