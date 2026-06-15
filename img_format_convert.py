from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif


DEMJIH_FORMATUUD = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def convert_image(source_path: Path, destination_dir: Path) -> Path:
    """Convert one image to RGB JPEG while preserving EXIF orientation."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        output_path = destination_dir / f"{source_path.stem}.jpeg"
        img.save(output_path, format="JPEG", quality=95, optimize=True)

    return output_path


def iter_input_images(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in DEMJIH_FORMATUUD:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="./assets directory доторх зургуудыг RGB JPEG болгох."
    )
    parser.add_argument("input_dir", nargs="?", default="raw_assets", help="Эх raw зургуудыг хадгалах directory(default: raw_assets)")
    parser.add_argument("output_dir", nargs="?", default="assets", help="Хувиргасан JPEG зургуудыг хадгалах directory (default: assets)")   
    args = parser.parse_args()

    pillow_heif.register_heif_opener()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Оролтын directory байхгүй байна: {input_dir}")

    converted = 0
    for source_path in iter_input_images(input_dir):
        destination_path = convert_image(source_path, output_dir)
        print(f"{source_path.name} -> {destination_path}")
        converted += 1

    print(f"{converted} зургийг(s) {output_dir.resolve()} болгон хувиргав.")


if __name__ == "__main__":
    main()