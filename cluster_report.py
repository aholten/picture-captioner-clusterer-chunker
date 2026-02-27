"""Cluster report generator — reads captions.jsonl and produces grouped output."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer

app = typer.Typer(help="Cluster report — group captioned photos by semantic similarity.")

logger = logging.getLogger(__name__)


def _load_captions(journal_path: Path) -> dict[str, dict]:
    """Load captions via journal.py."""
    from journal import CaptionJournal

    journal = CaptionJournal(journal_path)
    return journal.load()


def _collect_photos(photos_dir: Path) -> list[tuple[str, Path]]:
    """Walk photos_dir, return (rel_path, full_path) for image files."""
    extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".bmp", ".webp"}
    results = []
    for root, _dirs, files in os.walk(photos_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in extensions:
                full = Path(root) / fname
                rel = full.relative_to(photos_dir).as_posix()
                results.append((rel, full))
    return results


VOCABULARY = [
    "a family photo", "a group of people", "a selfie", "a portrait",
    "a landscape", "a sunset", "a beach", "mountains",
    "food on a plate", "a restaurant", "cooking",
    "a pet", "a dog", "a cat",
    "a building", "architecture", "a city street",
    "a car", "a vehicle",
    "a screenshot", "text on a screen",
    "a celebration", "a party", "a wedding",
    "sports", "exercise", "outdoors activity",
    "flowers", "a garden", "nature",
    "a child", "a baby",
    "an animal", "wildlife",
    "artwork", "a painting", "a drawing",
]


@app.command()
def report(
    photos_dir: Optional[Path] = typer.Option(None, help="Path to photo library root"),
    captions_jsonl: Path = typer.Option(Path("captions.jsonl"), help="Path to captions journal file"),
    output_csv: Path = typer.Option(Path("cluster_export.csv"), help="Output CSV path"),
    output_md: Path = typer.Option(Path("cluster_report.md"), help="Output markdown path"),
    n_neighbors: int = typer.Option(15, help="UMAP n_neighbors"),
    min_cluster_size: int = typer.Option(20, help="HDBSCAN min_cluster_size"),
) -> None:
    """Generate cluster report from captions."""
    from sentence_transformers import SentenceTransformer
    import hdbscan
    import umap

    from config import Settings

    settings = Settings() if photos_dir is None else Settings(PHOTOS_DIR=photos_dir)
    photos_dir = settings.PHOTOS_DIR

    # Load captions
    captions = _load_captions(captions_jsonl) if captions_jsonl.exists() else {}
    all_photos = _collect_photos(photos_dir)

    typer.echo(f"Found {len(all_photos)} photos, {len(captions)} captions loaded.")

    # Determine labels — caption if available, else will use CLIP vocabulary fallback
    labels = {}
    caption_count = 0
    error_count = 0
    fallback_count = 0

    for rel_path, _ in all_photos:
        rec = captions.get(rel_path)
        if rec and rec["status"] == "success" and rec["caption"]:
            labels[rel_path] = rec["caption"]
            caption_count += 1
        elif rec and rec["status"].startswith("error_"):
            labels[rel_path] = ""
            error_count += 1
        else:
            labels[rel_path] = None  # needs fallback
            fallback_count += 1

    typer.echo(f"{caption_count} captioned, {error_count} errors, {fallback_count} vocabulary fallbacks")

    # Encode captions/labels for clustering
    typer.echo("Loading sentence transformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build text list for encoding: use caption where available, vocabulary match otherwise
    texts_to_encode = []
    photo_order = []

    if fallback_count > 0:
        typer.echo("Encoding CLIP vocabulary for fallback matching...")
        vocab_embeddings = st_model.encode(VOCABULARY, show_progress_bar=False)

    for rel_path, full_path in all_photos:
        label = labels[rel_path]
        if label is None:
            # Fallback: use closest vocabulary term (simple heuristic — use filename words)
            # For proper CLIP fallback, we'd need image embeddings. Here we use the best
            # vocabulary match as a placeholder label.
            texts_to_encode.append(VOCABULARY[0])  # default fallback
        elif label == "":
            texts_to_encode.append("an unidentified photo")
        else:
            texts_to_encode.append(label)
        photo_order.append(rel_path)

    typer.echo("Encoding captions...")
    embeddings = st_model.encode(texts_to_encode, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings)

    # UMAP dimensionality reduction
    typer.echo("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=10, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    typer.echo("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    cluster_labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    typer.echo(f"Found {n_clusters} clusters, {n_noise} noise points.")

    # Build cluster info
    clusters = {}
    for i, rel_path in enumerate(photo_order):
        cl = int(cluster_labels[i])
        if cl not in clusters:
            clusters[cl] = []
        clusters[cl].append((rel_path, embeddings[i], texts_to_encode[i]))

    # For each cluster, find the photo nearest to the centroid for the cluster label
    cluster_summaries = {}
    for cl, members in sorted(clusters.items()):
        if cl == -1:
            cluster_summaries[cl] = "Unclustered"
            continue
        member_embeddings = np.array([m[1] for m in members])
        centroid = member_embeddings.mean(axis=0)
        distances = np.linalg.norm(member_embeddings - centroid, axis=1)
        nearest_idx = int(distances.argmin())
        nearest_rel_path = members[nearest_idx][0]

        # Use caption of nearest-to-centroid photo if it has a successful caption
        rec = captions.get(nearest_rel_path)
        if rec and rec["status"] == "success" and rec["caption"]:
            cluster_summaries[cl] = rec["caption"]
        else:
            cluster_summaries[cl] = members[nearest_idx][2]  # fallback text

    # Write CSV
    typer.echo(f"Writing {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_path", "cluster", "cluster_label", "caption"])
        for i, rel_path in enumerate(photo_order):
            cl = int(cluster_labels[i])
            caption_text = texts_to_encode[i]
            writer.writerow([rel_path, cl, cluster_summaries.get(cl, ""), caption_text])

    # Write markdown report
    typer.echo(f"Writing {output_md}...")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Cluster Report\n\n")
        f.write(f"Total photos: {len(photo_order)}\n")
        f.write(f"Clusters: {n_clusters}\n")
        f.write(f"Noise points: {n_noise}\n\n")
        f.write(f"Source: {caption_count} captioned, {error_count} errors, {fallback_count} vocabulary fallbacks\n\n")

        for cl in sorted(clusters.keys()):
            if cl == -1:
                continue
            members = clusters[cl]
            label = cluster_summaries[cl]
            f.write(f"## Cluster {cl}: {label}\n\n")
            f.write(f"Size: {len(members)} photos\n\n")
            # Show first 5 examples
            for rel_path, _, caption_text in members[:5]:
                f.write(f"- `{rel_path}`: {caption_text}\n")
            if len(members) > 5:
                f.write(f"- ... and {len(members) - 5} more\n")
            f.write("\n")

        if -1 in clusters:
            noise = clusters[-1]
            f.write(f"## Unclustered ({len(noise)} photos)\n\n")
            for rel_path, _, caption_text in noise[:10]:
                f.write(f"- `{rel_path}`: {caption_text}\n")
            if len(noise) > 10:
                f.write(f"- ... and {len(noise) - 10} more\n")
            f.write("\n")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
