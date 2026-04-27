import cv2
from pathlib import Path
import random
import numpy as np

CARDS_DIR = Path("C:/dataset research/assets/cards") 
TEXTURE_DIR = Path("C:/dataset research/assets/texture")
DATASET_DIR = Path("C:/dataset research/dataset_02_overlays")

card_paths = list(CARDS_DIR.glob("*.png"))
bg_paths = list(TEXTURE_DIR.glob("*.jpg"))

class_names = sorted([p.stem for p in card_paths])
class_to_id = {name: i for i, name in enumerate(class_names)}

IMG_SIZE = 768

yaml_path = DATASET_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {DATASET_DIR.as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")
    for i, name in enumerate(class_names):
        f.write(f"  {i}: {name}\n")

cards = {p.stem: cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in card_paths}
backgrounds = [cv2.imread(str(p)) for p in bg_paths]

def blend(bg, card, x, y):
    h, w = card.shape[:2]
    roi = bg[y:y+h, x:x+w].astype(np.float32)
    rgb = card[:, :, :3].astype(np.float32)
    alpha = (card[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    bg[y:y+h, x:x+w] = np.clip(alpha * rgb + (1 - alpha) * roi, 0, 255).astype(np.uint8)

def alpha_bbox(card, x, y):
    alpha = card[:, :, 3]
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0:
        return None

    x1 = xs.min() + x
    y1 = ys.min() + y
    x2 = xs.max() + 1 + x
    y2 = ys.max() + 1 + y

    return x1, y1, x2, y2

def yolo_label(class_id, box, img_size=IMG_SIZE):
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2

    return f"{class_id} {cx/img_size:.6f} {cy/img_size:.6f} {bw/img_size:.6f} {bh/img_size:.6f}"

def generate_random_positions(count=3, min_dist=40):
    positions = []

    while len(positions) < count:
        x = random.randint(0, 528)
        y = random.randint(0, 434)

        valid = True

        for px, py in positions:
            distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5

            if distance < min_dist:
                valid = False
                break

        if valid:
            positions.append((x, y))

    return positions

def generate_one_image():
    bg = random.choice(backgrounds).copy()
    bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

    chosen_cards = random.sample(class_names, 3)

    positions = generate_random_positions()

    labels = []

    for name, (x, y) in zip(chosen_cards, positions):
        card = cards[name]

        blend(bg, card, x, y)

        box = alpha_bbox(card, x, y)
        if box is not None:
            labels.append(yolo_label(class_to_id[name], box))

    return bg, labels

def save_split(split, count, start_idx=0):
    for i in range(count):
        img, labels = generate_one_image()
        idx = start_idx + i

        img_path = DATASET_DIR / "images" / split / f"img_{idx:05d}.jpg"
        txt_path = DATASET_DIR / "labels" / split / f"img_{idx:05d}.txt"

        cv2.imwrite(str(img_path), img)

        with open(txt_path, "w") as f:
            for line in labels:
                f.write(line + "\n")

save_split("train", 4000, 0)
save_split("val", 500, 4000)

print("dataset_01_control elkészült.")