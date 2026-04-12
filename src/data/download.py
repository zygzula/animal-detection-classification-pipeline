import shutil
from pathlib import Path
from typing import Any, Dict, List

import requests

import src.config as config


def _build_image_url(item: Dict[str, Any]) -> str:
    file_name = item["file_name"]

    return f"{config.BASE_IMAGE_URL}/{file_name}"


def _download_image(
        session: requests.Session,
        url: str,
        target_path: Path,
        timeout: int = config.REQUEST_TIMEOUT,
        verbose: bool = config.VERBOSE,
) -> bool:
    if target_path.exists():
        return True

    try:
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        if verbose:
            print(f"Request failed for {url}: {exc}.")

        return False

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        if verbose:
            print(f"Not an image: {url} (Content-Type: {content_type}).")

        return False

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)

        return True

    except OSError as exc:
        if verbose:
            print(f"Could not save {target_path}: {exc}.")

        return False


def delete_temporary_images(download_dir: Path) -> None:
    if download_dir.exists() and download_dir.is_dir():
        shutil.rmtree(download_dir)


def prepare_batch(
        manifest_items: List[Dict[str, Any]],
        download_dir: Path,
        batch_number: int,
        batch_size: int,
        session: requests.Session,
        timeout: int = config.REQUEST_TIMEOUT,
        verbose: bool = config.VERBOSE,
) -> List[Dict[str, Any]]:
    start = batch_number * batch_size
    end = start + batch_size
    source_items = manifest_items[start:end]

    downloaded_items: List[Dict[str, Any]] = []
    for i, item in enumerate(source_items):
        image_url = _build_image_url(item)

        original_name = item.get("file_name", f"image_{start + i}.jpg")
        target_path = download_dir / original_name

        success = _download_image(
            session=session,
            url=image_url,
            target_path=target_path,
            timeout=timeout,
            verbose=verbose,
        )

        if not success:
            continue

        downloaded_items.append(
            {
                **item,
                "image_url": image_url,
                "local_path": str(target_path),
                "local_filename": target_path.name,
            }
        )

    return downloaded_items
