import cv2
from pathlib import Path
import random
import numpy as np

CARDS_DIR = Path("C:/dataset research/assets/cards") 
TEXTURE_DIR = Path("C:/dataset research/assets/texture")
DATASET_DIR = Path("C:/dataset research/dataset_03_perspectives")

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

def resize_card(card):
    scale = random.uniform(0.8, 1.2)
    h, w = card.shape[:2]
    return cv2.resize(card, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def rotate_card(card):
    angle = random.uniform(-90, 90)

    h, w = card.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    rotated_w = int(h * sin + w * cos)
    rotated_h = int(h * cos + w * sin)

    M[0, 2] += rotated_w / 2 - center[0]
    M[1, 2] += rotated_h / 2 - center[1]

    return cv2.warpAffine(
        card,
        M,
        (rotated_w, rotated_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

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

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def box_distance(box1, box2):
    cx1, cy1 = box_center(box1)
    cx2, cy2 = box_center(box2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def generate_random_positions(resized_cards, min_dist=40):
    positions = []
    boxes = []

    for card in resized_cards:
        box = alpha_bbox(card, 0, 0)

        if box is None:
            positions.append((0, 0))
            continue

        x1, y1, x2, y2 = box

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        while True:
            x = random.randint(0, 738 - bbox_width)
            y = random.randint(0, 738 - bbox_height)

            placed_box = alpha_bbox(card, x, y)

            valid = True

            for existing_box in boxes:
                if box_distance(placed_box, existing_box) < min_dist:
                    valid = False
                    break

            if valid:
                positions.append((x, y))
                boxes.append(placed_box)
                break

    return positions

def apply_homography(img, boxes):
    src = np.float32([
        [0, 0],
        [IMG_SIZE, 0],
        [IMG_SIZE, IMG_SIZE],
        [0, IMG_SIZE]
    ])

    shift = IMG_SIZE // 4
    dst = src.copy()

    side = random.choice(["left", "right", "top", "bottom"])

    if side == "right":
        dst[1] = [IMG_SIZE, random.randint(0, shift)]
        dst[2] = [IMG_SIZE, random.randint(IMG_SIZE - shift, IMG_SIZE)]

    elif side == "left":
        dst[0] = [0, random.randint(0, shift)]
        dst[3] = [0, random.randint(IMG_SIZE - shift, IMG_SIZE)]

    elif side == "top":
        dst[0] = [random.randint(0, shift), 0]
        dst[1] = [random.randint(IMG_SIZE - shift, IMG_SIZE), 0]

    elif side == "bottom":
        dst[3] = [random.randint(0, shift), IMG_SIZE]
        dst[2] = [random.randint(IMG_SIZE - shift, IMG_SIZE), IMG_SIZE]

    H = cv2.getPerspectiveTransform(src, dst)

    warped_img = cv2.warpPerspective(img, H, (IMG_SIZE, IMG_SIZE))

    warped_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box

        points = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]).reshape(-1, 1, 2)

        warped_points = cv2.perspectiveTransform(points, H).reshape(-1, 2)

        xs = warped_points[:, 0]
        ys = warped_points[:, 1]

        new_box = (
            max(0, min(IMG_SIZE, xs.min())),
            max(0, min(IMG_SIZE, ys.min())),
            max(0, min(IMG_SIZE, xs.max())),
            max(0, min(IMG_SIZE, ys.max()))
        )

        warped_boxes.append(new_box)

    return warped_img, warped_boxes

def generate_one_image():
    bg = random.choice(backgrounds).copy()
    bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

    chosen_cards = random.sample(class_names, 3)

    resized_cards = []

    for name in chosen_cards:
        card = cards[name]
        card = resize_card(card)
        card = rotate_card(card)
        resized_cards.append(card)

    positions = generate_random_positions(resized_cards)

    boxes = []

    for name, card, (x, y) in zip(chosen_cards, resized_cards, positions):
        blend(bg, card, x, y)

        box = alpha_bbox(card, x, y)
        boxes.append(box)

    bg, boxes = apply_homography(bg, boxes)

    labels = []

    for name, box in zip(chosen_cards, boxes):
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

print("dataset_03_perspectives elkészült.")