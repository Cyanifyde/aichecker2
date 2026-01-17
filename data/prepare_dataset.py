import argparse
import csv
import hashlib
import io
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator

from PIL import Image

from data.transforms import LetterboxConfig, letterbox_512


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

logger = logging.getLogger(__name__)


def iter_images(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def normalize_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as img:
        converted = img.convert("RGB")
    config = LetterboxConfig(size=size, pad_mode="reflect")
    return letterbox_512(converted, config, interpolation=Image.Resampling.LANCZOS)


def encode_png(img: Image.Image, *, optimize: bool = True) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=optimize)
    return buffer.getvalue()


def write_image(bytes_png: bytes, out_path: Path) -> None:
    out_path.write_bytes(bytes_png)


def _process_one_image(
    path: Path,
    label: int,
    size: int,
    *,
    png_optimize: bool,
) -> tuple[Path, int, str | None, bytes | None, str | None]:
    try:
        normalized = normalize_image(path, size)
        png_bytes = encode_png(normalized, optimize=png_optimize)
        normalized.close()
        content_hash = hashlib.sha256(png_bytes).hexdigest()
        return path, label, content_hash, png_bytes, None
    except Exception as exc:  # noqa: BLE001 - data pipelines should be resilient to bad files
        return path, label, None, None, f"{type(exc).__name__}: {exc}"


def _iter_processed_images(
    executor: ThreadPoolExecutor,
    source_dir: Path,
    label: int,
    size: int,
    *,
    png_optimize: bool,
    max_in_flight: int,
) -> Iterator[tuple[Path, int, str | None, bytes | None, str | None]]:
    in_flight: set[Future] = set()
    for path in iter_images(source_dir):
        in_flight.add(
            executor.submit(_process_one_image, path, label, size, png_optimize=png_optimize)
        )
        if len(in_flight) >= max_in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()

    while in_flight:
        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()


def _setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize AI/real image folders into a training-ready format.")
    parser.add_argument("--ai-dir", type=Path, default=Path("ai"))
    parser.add_argument("--real-dir", type=Path, default=Path("real"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--manifest-name", type=str, default="manifest.csv")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker threads. 0 = auto.")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-every", type=int, default=250, help="Log progress every N processed images.")
    parser.add_argument(
        "--no-png-optimize",
        action="store_true",
        help="Disable PNG optimization (faster, larger files).",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source_path"])

        workers = args.workers if args.workers > 0 else max(1, os.cpu_count() or 4)
        max_in_flight = max(16, workers * 4)
        png_optimize = not args.no_png_optimize

        logger.info(
            "Preparing dataset (workers=%d, size=%d, png_optimize=%s) -> %s",
            workers,
            args.size,
            png_optimize,
            manifest_path.as_posix(),
        )

        seen_hashes: set[str] = set()
        processed = 0
        written = 0
        duplicates = 0
        errors = 0
        start = time.perf_counter()

        def handle_result(result: tuple[Path, int, str | None, bytes | None, str | None]) -> None:
            nonlocal processed, written, duplicates, errors
            path, label, content_hash, png_bytes, error = result
            processed += 1
            if error is not None:
                errors += 1
                logger.warning("Failed to process %s (%s)", path.as_posix(), error)
                return
            assert content_hash is not None
            assert png_bytes is not None
            if content_hash in seen_hashes:
                duplicates += 1
                return
            seen_hashes.add(content_hash)
            filename = f"{content_hash}.png"
            class_dir = output_dir / ("ai" if label == 1 else "real")
            out_path = class_dir / filename
            write_image(png_bytes, out_path)
            rel_path = out_path.relative_to(output_dir)
            writer.writerow([str(rel_path), label, path.as_posix()])
            written += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                elapsed = max(1e-6, time.perf_counter() - start)
                rate = processed / elapsed
                logger.info(
                    "Progress: processed=%d kept=%d dupes=%d errors=%d (%.1f img/s)",
                    processed,
                    written,
                    duplicates,
                    errors,
                    rate,
                )

        (output_dir / "ai").mkdir(parents=True, exist_ok=True)
        (output_dir / "real").mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in _iter_processed_images(
                executor,
                args.ai_dir,
                1,
                args.size,
                png_optimize=png_optimize,
                max_in_flight=max_in_flight,
            ):
                handle_result(result)

            for result in _iter_processed_images(
                executor,
                args.real_dir,
                0,
                args.size,
                png_optimize=png_optimize,
                max_in_flight=max_in_flight,
            ):
                handle_result(result)

        elapsed = max(1e-6, time.perf_counter() - start)
        logger.info(
            "Done: processed=%d kept=%d dupes=%d errors=%d in %.1fs (%.1f img/s)",
            processed,
            written,
            duplicates,
            errors,
            elapsed,
            processed / elapsed,
        )

    logger.info("Wrote %d items to %s", written, manifest_path.as_posix())


if __name__ == "__main__":
    main()
