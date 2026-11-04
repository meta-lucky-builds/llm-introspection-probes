"""Write a Colab cell that unpacks src/ and experiments/ from an embedded base64 zip.

No token goes into Colab, so the code travels by paste.

    python scripts/make_colab_bundle.py [--out dist/colab_cell.py]

Standard library only: this script needs neither torch nor the package.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INCLUDED_DIRS = ("src", "experiments")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "results"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
LINE_WIDTH = 120
TARGET_DIR = "/content/mechinterp"


def bundle_files(root: Path) -> list[Path]:
    files = []
    for directory in INCLUDED_DIRS:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
                continue
            relative = path.relative_to(root)
            if EXCLUDED_PARTS.intersection(relative.parts) or any(part.endswith(".egg-info") for part in relative.parts):
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_zip(root: Path) -> bytes:
    """A deterministic zip: fixed timestamps and sorted names, so equal trees give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in bundle_files(root):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def build_cell(root: Path) -> str:
    data = build_zip(root)
    digest = hashlib.sha256(data).hexdigest()
    blob = base64.b64encode(data).decode("ascii")
    wrapped = "\n".join(blob[start : start + LINE_WIDTH] for start in range(0, len(blob), LINE_WIDTH))
    return f'''# E1 code bundle, sha256 {digest}
import base64, io, os, sys, zipfile
BUNDLE = """
{wrapped}
"""
os.makedirs("{TARGET_DIR}", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall("{TARGET_DIR}")
if "{TARGET_DIR}/src" not in sys.path:
    sys.path.insert(0, "{TARGET_DIR}/src")
os.chdir("{TARGET_DIR}")
print("Unpacked bundle {digest[:12]} to {TARGET_DIR}. Restart the runtime first if replacing an imported version.")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "dist" / "colab_cell.py"))
    args = parser.parse_args()
    cell = build_cell(REPO_ROOT)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cell, encoding="utf-8")
    print(f"Wrote {out} ({len(cell)} characters). Paste its contents into one Colab cell.")


if __name__ == "__main__":
    main()
