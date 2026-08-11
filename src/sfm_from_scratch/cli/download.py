"""Download a Middlebury ring dataset into data/."""

import urllib.request
import zipfile

import click
from loguru import logger

from sfm_from_scratch.config import DATA_DIR, DATASETS


@click.command()
@click.option(
    "--dataset",
    "name",
    default="templeRing",
    type=click.Choice(sorted(DATASETS)),
    show_default=True,
)
def download(name: str) -> None:
    """Fetch and extract a Middlebury multi-view dataset (Seitz et al., CVPR 2006)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name
    if target.exists() and any(target.glob("*.png")):
        logger.info(f"{target} already populated — nothing to do")
        return
    zip_path = DATA_DIR / f"{name}.zip"
    if not zip_path.exists():
        url = DATASETS[name]
        logger.info(f"downloading {url}")
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — fixed https URL
        logger.info(f"saved {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    n_imgs = len(list(target.glob("*.png")))
    logger.info(f"extracted {n_imgs} images to {target}")
    if n_imgs == 0:
        raise click.ClickException(f"extraction produced no images under {target}")


if __name__ == "__main__":
    download()
