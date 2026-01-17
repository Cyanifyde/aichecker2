import argparse
import csv
import hashlib
import io
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from PIL import Image

from data.transforms import LetterboxConfig, letterbox_512


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def iter_images(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def normalize_image(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    config = LetterboxConfig(size=size, pad_mode="reflect")
    return letterbox_512(img, config, interpolation=Image.Resampling.LANCZOS)


def encode_png(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def write_image(bytes_png: bytes, out_path: Path) -> None:
    out_path.write_bytes(bytes_png)


def build_manifest_rows(
    source_dir: Path,
    output_dir: Path,
    label: int,
    size: int,
    seen_hashes: dict,
) -> Iterable[Tuple[str, int, str]]:
    for path in iter_images(source_dir):
        normalized = normalize_image(path, size)
        png_bytes = encode_png(normalized)
        content_hash = hashlib.sha256(png_bytes).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes[content_hash] = path
        filename = f"{content_hash}.png"
        class_dir = output_dir / ("ai" if label == 1 else "real")
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / filename
        write_image(png_bytes, out_path)
        rel_path = out_path.relative_to(output_dir)
        yield str(rel_path), label, path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize AI/real image folders into a training-ready format.")
    parser.add_argument("--ai-dir", type=Path, default=Path("ai"))
    parser.add_argument("--real-dir", type=Path, default=Path("real"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--manifest-name", type=str, default="manifest.csv")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    seen_hashes: dict = {}
    rows = []
    rows.extend(build_manifest_rows(args.ai_dir, output_dir, 1, args.size, seen_hashes))
    rows.extend(build_manifest_rows(args.real_dir, output_dir, 0, args.size, seen_hashes))

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source_path"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} items to {manifest_path}")


if __name__ == "__main__":
    main()
