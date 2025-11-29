import os
import csv
import time
import random
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config_mushrooms import BASE_OUTPUT_DIR, MUSHROOM_CLASSES, CLASS_URLS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Parámetros de reintentos / backoff
MAX_RETRIES = 3          # nº máximo de reintentos
BACKOFF_BASE = 1.0       # segundos: 1, 2, 4, ...


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_soup(url: str, timeout: int = 15) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[ERROR] Al obtener {url} (intento {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                return None
            sleep_time = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"[INFO]  Esperando {sleep_time:.2f}s antes de reintentar {url}")
            time.sleep(sleep_time)


def is_valid_image_url(img_url: str) -> bool:
    if not img_url:
        return False
    parsed = urllib.parse.urlparse(img_url)
    if parsed.scheme not in ("http", "https"):
        return False
    _, ext = os.path.splitext(parsed.path.lower())
    return ext in VALID_EXTENSIONS


def normalize_url(path: str, base_url: str) -> str:
    return urllib.parse.urljoin(base_url, path)


def download_image(img_url: str, dest_dir: Path, prefix: str) -> Path | None:
    parsed = urllib.parse.urlparse(img_url)
    _, ext = os.path.splitext(parsed.path.lower())
    if ext not in VALID_EXTENSIONS:
        ext = ".jpg"

    filename = f"{prefix}_{int(time.time() * 1000)}_{random.randint(0, 9999)}{ext}"
    dest_path = dest_dir / filename

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(img_url, headers=HEADERS, timeout=20, stream=True)
            resp.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return dest_path

        except Exception as e:
            print(f"[ERROR] Descargar {img_url} (intento {attempt}/{MAX_RETRIES}): {e}")
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass

            if attempt == MAX_RETRIES:
                return None

            sleep_time = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"[INFO]  Esperando {sleep_time:.2f}s antes de reintentar {img_url}")
            time.sleep(sleep_time)


def is_relevant_img_tag(img, genus_filter: str) -> bool:
    """
    Devuelve True solo si el atributo alt contiene el nombre del género
    y NO contiene 'spore' ni 'basidia' (para excluir imágenes de esporas, etc.).
    """
    alt_text = (img.get("alt") or "").strip()
    if not alt_text:
        return False

    alt_lower = alt_text.lower()

    if genus_filter.lower() not in alt_lower:
        return False

    if "spore" in alt_lower or "basidia" in alt_lower:
        return False

    return True


def scrape_images_from_species_page(species_url: str, class_name: str,
                                    dest_dir: Path, csv_writer: csv.writer,
                                    max_imgs: int | None = None) -> int:
    print(f"[INFO]   Scrapeando especie: {species_url}")
    soup = get_soup(species_url)
    if soup is None:
        return 0

    if class_name == "agaricus":
        genus_filter = "Agaricus"
    elif class_name == "amanita":
        genus_filter = "Amanita"
    elif class_name == "boletus":
        genus_filter = "Boletus"
    elif class_name == "cortinarius":
        genus_filter = "Cortinarius"
    elif class_name == "entoloma":
        genus_filter = "Entoloma"
    elif class_name == "hygrocybe":
        genus_filter = "Hygrocybe"
    elif class_name == "lactarius":
        genus_filter = "Lactarius"
    elif class_name == "russula":
        genus_filter = "Russula"
    elif class_name == "suillus":
        genus_filter = "Suillus"
    else:
        genus_filter = class_name.capitalize()

    img_tags = soup.find_all("img")
    count = 0

    for img in img_tags:
        if max_imgs is not None and count >= max_imgs:
            break

        if not is_relevant_img_tag(img, genus_filter):
            continue

        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src:
            continue

        img_url = normalize_url(src, species_url)
        if not is_valid_image_url(img_url):
            continue

        img_path = download_image(img_url, dest_dir, class_name)
        if img_path is None:
            continue

        csv_writer.writerow([str(img_path), class_name, species_url, img_url])
        count += 1

        time.sleep(random.uniform(0.5, 1.2))

    print(f"[INFO]   {count} imágenes descargadas de {species_url}")
    return count


def find_species_links(family_url: str, genus_filter: str) -> list[str]:
    """
    En la página de familia/orden, busca enlaces a especies
    cuyo texto contenga el nombre del género (p.ej., 'Agaricus').
    """
    print(f"[INFO] Buscando especies de {genus_filter} en {family_url}")
    soup = get_soup(family_url)
    if soup is None:
        return []

    links = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip()
        if genus_filter.lower() in text.lower():
            species_url = normalize_url(a["href"], family_url)
            links.append(species_url)

    seen = set()
    unique_links = []
    for u in links:
        if u not in seen:
            seen.add(u)
            unique_links.append(u)

    print(f"[INFO]   Encontrados {len(unique_links)} enlaces de especie para {genus_filter}")
    return unique_links


def main():
    base_dir = ensure_dir(BASE_OUTPUT_DIR)
    csv_path = base_dir / "metadata.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath", "label", "source_page", "image_url"])

        for class_name in MUSHROOM_CLASSES:
            class_dir = ensure_dir(base_dir / class_name)

            urls = CLASS_URLS.get(class_name, [])
            if not urls:
                print(f"[WARN] No hay URLs configuradas para la clase {class_name}")
                continue

            if class_name == "agaricus":
                genus_filter = "Agaricus"
            elif class_name == "amanita":
                genus_filter = "Amanita"
            elif class_name == "boletus":
                genus_filter = "Boletus"
            elif class_name == "cortinarius":
                genus_filter = "Cortinarius"
            elif class_name == "entoloma":
                genus_filter = "Entoloma"
            elif class_name == "hygrocybe":
                genus_filter = "Hygrocybe"
            elif class_name == "lactarius":
                genus_filter = "Lactarius"
            elif class_name == "russula":
                genus_filter = "Russula"
            elif class_name == "suillus":
                genus_filter = "Suillus"
            else:
                genus_filter = class_name.capitalize()

            total_imgs = 0

            for family_url in urls:
                species_links = find_species_links(family_url, genus_filter)

                for species_url in species_links:
                    total_imgs += scrape_images_from_species_page(
                        species_url,
                        class_name,
                        class_dir,
                        writer,
                        max_imgs=None,
                    )
                    time.sleep(random.uniform(1.5, 3.0))

                time.sleep(random.uniform(2.0, 4.0))

            print(f"[INFO] Clase {class_name}: {total_imgs} imágenes descargadas.")

    print(f"[INFO] Scraping terminado. Metadatos en: {csv_path}")


if __name__ == "__main__":
    main()
