# download_pmo.py
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import tarfile


def main():
    URL = "https://figshare.com/ndownloader/files/35994221"
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    OUT_DIR = PROJECT_ROOT / "data" / "results" / "pmo_baseline"
    DOWNLOAD = OUT_DIR / "pmo_download"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading...")
    urlretrieve(URL, DOWNLOAD)

    head = DOWNLOAD.read_bytes()[:8]

    print("Unpacking...")
    if head.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(DOWNLOAD) as z:
            z.extractall(OUT_DIR)
    else:
        with tarfile.open(DOWNLOAD, "r:*") as t:
            t.extractall(OUT_DIR)

    DOWNLOAD.unlink()  # delete downloaded bundle

    # Remove macOS AppleDouble metadata files (._* files)
    for f in OUT_DIR.glob("._*"):
        f.unlink()

    print(f"Done. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()