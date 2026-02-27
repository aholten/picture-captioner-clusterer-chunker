"""CLI orchestrator for the picture captioning pipeline."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import typer
from tqdm import tqdm

from backends import load_backend
from backends.base import CorruptImageError
from config import Settings
from image_loader import load_image
from journal import CaptionJournal

app = typer.Typer(help="Picture Captioner — generate per-photo captions with pluggable backends.")

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
    }
)

DEFAULT_PROMPT = (
    "Describe this photo in one or two sentences. Focus on the main subject, setting, and activity."
)

SUBMIT_BATCH = 500  # bound in-flight futures


def _setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("caption")
    logger.setLevel(logging.DEBUG)

    # Console handler — human-readable
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    # File handler — JSON lines
    from pythonjsonlogger.json import JsonFormatter

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonFormatter(timestamp=True))
    logger.addHandler(fh)

    return logger


def _collect_photos(photos_dir: Path) -> list[tuple[str, Path]]:
    """Walk photos_dir, return list of (rel_path, full_path) for image files."""
    results = []
    for root, _dirs, files in os.walk(photos_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                full = Path(root) / fname
                rel = full.relative_to(photos_dir).as_posix()
                results.append((rel, full))
    return results


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


@app.command()
def run(
    backend: str = typer.Option("mock", help="Backend name (see docs for options)"),
    model: str = typer.Option("mock", help="Model identifier for the backend"),
    photos_dir: Path | None = typer.Option(None, help="Path to photo library root"),
    captions_jsonl: Path = typer.Option(Path("captions.jsonl"), help="Captions journal file"),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Caption prompt"),
    max_workers: int = typer.Option(1, help="Concurrent workers (1 for local, 8+ for API)"),
    limit: int = typer.Option(0, help="Max photos to process (0 = unlimited)"),
    restart_every: int = typer.Option(0, help="Exit code 2 after N photos (0 = disabled)"),
    retry_status: str | None = typer.Option(None, help="Statuses to retry, e.g. error_api"),
    force: bool = typer.Option(False, help="Reprocess everything from scratch"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate images with MockBackend"),
) -> None:
    """Run the captioning pipeline."""
    log_file = captions_jsonl.parent / "caption.log"
    logger = _setup_logging(log_file)

    settings = Settings() if photos_dir is None else Settings(PHOTOS_DIR=photos_dir)
    photos_dir = settings.PHOTOS_DIR

    if not photos_dir.is_dir():
        logger.error("PHOTOS_DIR does not exist: %s", photos_dir)
        raise typer.Exit(1)

    # Override backend to mock for dry-run
    actual_backend_name = "mock" if dry_run else backend
    actual_model = "dry-run" if dry_run else model

    be = load_backend(actual_backend_name, actual_model, settings)

    # Clamp max_workers to 1 for local backend
    if backend == "local" and max_workers > 1:
        logger.info("Local backend is single-threaded (GPU constraint). Ignoring --max-workers.")
        max_workers = 1

    # Load journal
    journal = CaptionJournal(captions_jsonl)
    if force:
        # Start fresh — don't load existing records
        logger.info("--force: ignoring existing captions.jsonl")
    else:
        retry_statuses = set(retry_status.split(",")) if retry_status else frozenset()
        journal.load(retry_statuses=retry_statuses)
        if journal._records:
            logger.info(journal.summary())

    # Collect photos
    logger.info("Scanning %s for images...", photos_dir)
    all_photos = _collect_photos(photos_dir)
    logger.info("Found %d total images.", len(all_photos))

    # Filter to pending work
    pending = [(rp, fp) for rp, fp in all_photos if not journal.is_done(rp)]
    if limit > 0:
        pending = pending[:limit]
    logger.info("%d images to process this run.", len(pending))

    if not pending:
        logger.info("Nothing to do.")
        journal.close()
        raise typer.Exit(0)

    # Process
    processed = 0

    def process_one(rel_path: str, full_path: Path) -> None:
        try:
            image = load_image(full_path)
            caption = be.caption(image)
            journal.write(rel_path, caption, "success")
        except CorruptImageError as e:
            journal.write(rel_path, None, "error_corrupt")
            logger.warning("error_corrupt: %s — %s", rel_path, e)
        except Exception as e:
            status = "error_api" if "api" in backend else "error_model"
            journal.write(rel_path, None, status)
            logger.warning("%s: %s — %s", status, rel_path, e)

    with tqdm(total=len(pending), desc="Captioning", unit="img") as pbar:
        if max_workers <= 1:
            for rp, fp in pending:
                process_one(rp, fp)
                processed += 1
                pbar.update(1)
                if restart_every > 0 and processed >= restart_every:
                    break
        else:
            done = False
            for batch in _batched(pending, SUBMIT_BATCH):
                if done:
                    break
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(process_one, rp, fp): rp for rp, fp in batch}
                    for future in as_completed(futures):
                        future.result()
                        processed += 1
                        pbar.update(1)
                        if restart_every > 0 and processed >= restart_every:
                            done = True
                            break

    journal.close()

    # Exit code: 0 = all done, 2 = batch complete (more to do)
    remaining = len(pending) - processed
    if restart_every > 0 and remaining > 0:
        logger.info("Batch of %d done. %d remaining.", processed, remaining)
        raise typer.Exit(2)

    logger.info("Done. Processed %d images.", processed)


@app.command()
def estimate(
    backend: str = typer.Option("openai", help="Backend name for cost estimation"),
    model: str = typer.Option("gpt-4o-mini", help="Model identifier"),
    photos_dir: Path | None = typer.Option(None, help="Path to photo library root"),
    captions_jsonl: Path = typer.Option(Path("captions.jsonl"), help="Captions journal file"),
    max_workers: int = typer.Option(8, help="Assumed concurrent workers for time estimate"),
) -> None:
    """Estimate time and cost without running inference."""
    settings = Settings() if photos_dir is None else Settings(PHOTOS_DIR=photos_dir)
    photos_dir = settings.PHOTOS_DIR

    all_photos = _collect_photos(photos_dir)
    total = len(all_photos)

    journal = CaptionJournal(captions_jsonl)
    journal.load()
    already_done = sum(1 for rp, _ in all_photos if journal.is_done(rp))
    remaining = total - already_done

    # Rough estimates per model: (input_cost, output_cost, seconds) per image
    _default = (0.0002, 0.00006, 1.0)
    cost_estimates = {
        "gpt-4o-mini": (0.00015, 0.00005, 0.8),
        "gpt-4o": (0.00032, 0.00012, 1.0),
        "gemini-1.5-flash": (0.00008, 0.00003, 0.5),
        "gemini-2.0-flash": (0.00012, 0.00005, 0.6),
        "claude-haiku-4-5": (0.0012, 0.0004, 1.0),
    }

    inp, outp, secs = cost_estimates.get(model, _default)
    input_cost = remaining * inp
    output_cost = remaining * outp
    total_cost = input_cost + output_cost
    mins = (remaining * secs) / max_workers / 60

    typer.echo(f"Photos remaining:  {remaining:,} / {total:,} ({already_done:,} already done)")
    typer.echo(f"Backend:           {backend} / {model}")
    typer.echo(
        f"Estimated cost:    ~${input_cost:.2f} (input) "
        f"+ ~${output_cost:.2f} (output) = ~${total_cost:.2f}"
    )
    typer.echo(f"Estimated time:    ~{mins:.0f} min at {max_workers} workers")


@app.command()
def stats(
    captions_jsonl: Path = typer.Option(Path("captions.jsonl"), help="Captions journal file"),
) -> None:
    """Print summary of an existing captions.jsonl."""
    journal = CaptionJournal(captions_jsonl)
    records = journal.load()

    if not records:
        typer.echo("No records found.")
        raise typer.Exit(0)

    from collections import Counter

    counts = Counter(rec["status"] for rec in records.values())
    total = sum(counts.values())

    typer.echo(f"Total records: {total:,}")
    for status, count in sorted(counts.items()):
        typer.echo(f"  {status}: {count:,}")


if __name__ == "__main__":
    app()
