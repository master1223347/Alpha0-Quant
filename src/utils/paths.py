"""Project path utilities."""

from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    """Find project root by walking upward until src/ and README.md are present."""
    current = Path(start) if start is not None else Path.cwd()
    current = current.resolve()

    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    return current


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return it as a resolved Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def ensure_parent(path: str | Path) -> Path:
    """Ensure parent directory for a file path exists."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def resolve_raw_root(raw_root: str | Path = "data/raw") -> Path:
    """Resolve raw root across supported layouts."""
    root = Path(raw_root)
    if root.exists():
        return root

    nested = root / "us"
    if nested.exists():
        return nested

    raise FileNotFoundError(f"Raw data directory not found at {root} or {nested}")
