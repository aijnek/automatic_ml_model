"""E2E検証用の合成データセット生成。

色付き図形の画像を生成し、ラベル = 図形の色（red/green/blue）として
data/images/ と data/annotations.csv、config.yaml を直接作る
（アノテーションアプリを経由せずにパイプラインを通しで検証するため）。

実行: uv run python scripts/make_dummy_dataset.py [--n 60]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (  # noqa: E402
    ANNOTATIONS_CSV,
    IMAGES_DIR,
    Config,
    save_config,
)

COLORS = {
    "red": [(220, 40, 40), (180, 30, 60), (240, 80, 70)],
    "green": [(40, 180, 60), (60, 160, 40), (30, 200, 90)],
    "blue": [(40, 70, 220), (30, 100, 200), (70, 60, 240)],
}
SHAPES = ("circle", "square", "triangle")
BACKGROUNDS = [(245, 245, 240), (230, 235, 245), (250, 240, 230), (210, 210, 210)]


def draw_image(color_name: str, rng: random.Random, size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), rng.choice(BACKGROUNDS))
    d = ImageDraw.Draw(img)
    fill = rng.choice(COLORS[color_name])
    shape = rng.choice(SHAPES)
    cx, cy = rng.randint(90, size - 90), rng.randint(90, size - 90)
    r = rng.randint(40, 80)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="生成枚数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    colors = list(COLORS)
    for i in range(args.n):
        color = colors[i % len(colors)]
        filename = f"dummy_{i:03d}.png"
        draw_image(color, rng).save(IMAGES_DIR / filename)
        rows.append(f"{filename},{color}")

    ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_CSV.write_text("filename,label\n" + "\n".join(rows) + "\n")

    save_config(
        Config(
            task_type="classification",
            description=(
                "単色の背景に単純な図形（円・四角・三角）が1つ描かれた合成画像。"
                "図形の色（red / green / blue）を分類する。"
            ),
            classes=colors,
            threshold=0.9,
        )
    )
    print(f"生成完了: {args.n}枚 → {IMAGES_DIR}")
    print(f"アノテーション: {ANNOTATIONS_CSV}")
    print("次: uv run python -m pipeline.split && uv run python -m pipeline.run_loop")


if __name__ == "__main__":
    main()
