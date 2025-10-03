"""Path utilities for file and directory management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .logging import get_logger

LOGGER = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Start from current file and go up until we find setup.py or pyproject.toml
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def get_artifacts_dir(subdir: Optional[str] = None) -> Path:
    """Get artifacts directory, optionally with subdirectory.
    
    Args:
        subdir: Optional subdirectory name
        
    Returns:
        Path to artifacts directory
    """
    root = get_project_root()
    artifacts = root / "artifacts"
    
    if subdir:
        artifacts = artifacts / subdir
    
    return ensure_dir(artifacts)


def get_config_dir() -> Path:
    """Get configuration directory.
    
    Returns:
        Path to configs directory
    """
    return get_project_root() / "configs"


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path string, handling relative paths.
    
    Args:
        path_str: Path string (can be relative or absolute)
        base_dir: Base directory for relative paths (defaults to project root)
        
    Returns:
        Resolved absolute path
    """
    path = Path(path_str)
    
    if path.is_absolute():
        return path
    
    if base_dir is None:
        base_dir = get_project_root()
    
    return base_dir / path


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename.
    
    Args:
        name: Original name
        
    Returns:
        Safe filename string
    """
    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = name
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(' .')
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"
    
    return safe_name


__all__ = [
    "ensure_dir",
    "get_project_root", 
    "get_artifacts_dir",
    "get_config_dir",
    "resolve_path",
    "safe_filename",
]