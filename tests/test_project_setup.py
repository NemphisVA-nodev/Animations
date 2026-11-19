"""Verify the Animations project structure and package imports.

These tests only confirm that the project is correctly initialized and that the
core and animation packages can be imported. No physics, rendering, or
exporting functionality is tested here, as none has been implemented yet.
"""

from importlib import import_module
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_PACKAGES = [
    "core",
    "core.physics",
    "core.rendering",
    "core.exporting",
]

ANIMATION_PACKAGES = [
    "animations",
    "animations.spaghettification",
    "animations.gravitational_lensing",
    "animations.orbital_mechanics",
    "animations.chaos",
]

EXPECTED_DIRECTORIES = [
    "animations",
    "animations/spaghettification",
    "animations/gravitational_lensing",
    "animations/orbital_mechanics",
    "animations/chaos",
    "core",
    "core/physics",
    "core/rendering",
    "core/exporting",
    "assets",
    "outputs",
    "tests",
    "docs",
]

EXPECTED_FILES = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    ".gitignore",
    "animations/__init__.py",
    "core/__init__.py",
    "tests/__init__.py",
    "outputs/.gitkeep",
    "assets/.gitkeep",
    "docs/architecture.md",
]


def _project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


@pytest.mark.parametrize("package", CORE_PACKAGES)
def test_core_package_imports(package: str) -> None:
    """Each core package can be imported without side effects."""
    import_module(package)


@pytest.mark.parametrize("package", ANIMATION_PACKAGES)
def test_animation_package_imports(package: str) -> None:
    """Each animation package can be imported without side effects."""
    import_module(package)


@pytest.mark.parametrize("relative", EXPECTED_DIRECTORIES)
def test_project_directory_exists(relative: str) -> None:
    """Each expected project directory exists."""
    assert _project_path(relative).is_dir(), f"missing directory: {relative}"


@pytest.mark.parametrize("relative", EXPECTED_FILES)
def test_project_file_exists(relative: str) -> None:
    """Each expected project file exists."""
    assert _project_path(relative).is_file(), f"missing file: {relative}"


def test_outputs_gitkeep_preserved() -> None:
    """The outputs directory is kept under version control via .gitkeep."""
    assert _project_path("outputs", ".gitkeep").is_file()


def test_project_root_has_metadata() -> None:
    """The project root contains pyproject.toml and README.md."""
    assert _project_path("pyproject.toml").is_file()
    assert _project_path("README.md").is_file()
