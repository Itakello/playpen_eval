import importlib
import inspect
from abc import ABC
from pathlib import Path
from typing import Type, TypeVar

from itakello_logging import ItakelloLogging
from pydantic import BaseModel

logger = ItakelloLogging().get_logger(__name__)
T = TypeVar("T", bound="CustomClass")


class CustomClass(ABC, BaseModel):

    @classmethod
    def get_all_subclasses(cls: Type[T]) -> dict[str, Type[T]]:
        """Get all subclasses in the same directory that inherit from this class."""
        current_dir, base_module_name = cls._get_module_path()
        if not current_dir:
            return {}

        py_files = cls._get_py_files(current_dir)
        return cls._load_subclasses(py_files, base_module_name)

    @classmethod
    def get_specific_subclasses(
        cls: Type[T], subclass_names: list[str]
    ) -> dict[str, Type[T]]:
        """Get specific subclasses by their names.

        Args:
            subclass_names: A list of subclass names to retrieve (case-insensitive)

        Returns:
            A dictionary containing the names and the subclasses if found
        """
        current_dir, base_module_name = cls._get_module_path()
        if not current_dir:
            return {}

        py_files = cls._get_py_files(current_dir)
        subclasses = cls._load_subclasses(py_files, base_module_name)

        found_subclasses = cls._filter_subclasses(subclasses, subclass_names)

        return found_subclasses

    @classmethod
    def get_subclass(cls: Type[T], subclass_name: str) -> Type[T]:
        """Get a specific subclass by its name"""
        return cls.get_specific_subclasses([subclass_name]).get(subclass_name)

    @classmethod
    def _filter_subclasses(
        cls, subclasses: dict[str, Type[T]], subclass_names: list[str]
    ) -> dict[str, Type[T]]:
        """Filter the subclasses based on the given names."""
        # Normalize subclass names to lowercase for case-insensitive matching
        subclass_names = [name.lower() for name in subclass_names]
        found_subclasses = {}

        for name, subclass in subclasses.items():
            if name.lower() in subclass_names:
                found_subclasses[name] = subclass

        missing_subclasses = set(subclass_names) - set(found_subclasses.keys())
        if missing_subclasses:
            logger.warning(f"Subclasses not found: {', '.join(missing_subclasses)}")

        return found_subclasses

    @classmethod
    def _get_module_path(cls) -> tuple[Path | None, str]:
        """Helper method to get the module path and base module name."""
        # Get the module where the subclass is defined
        subclass_module = inspect.getmodule(cls)

        # Handle case where module path cannot be determined
        module_file = getattr(subclass_module, "__file__", None)
        if not module_file:
            return None, ""

        current_dir = Path(str(module_file)).parent
        assert subclass_module
        base_module_name = subclass_module.__name__.rsplit(".", 1)[0]
        return current_dir, base_module_name

    @classmethod
    def _get_py_files(cls, current_dir: Path) -> list[Path]:
        """Helper method to get relevant Python files in the directory."""
        return [
            f
            for f in current_dir.glob("*.py")
            if f.is_file()
            and f.name != "__init__.py"
            and not f.name.startswith("custom_")
        ]

    @classmethod
    def _load_subclasses(
        cls: Type[T], py_files: list[Path], base_module_name: str
    ) -> dict[str, Type[T]]:
        """Helper method to load subclasses from Python files."""
        subclasses = {}
        for file in py_files:
            module_full_name = f"{base_module_name}.{file.stem}"
            module_name = file.stem.split("_")[0]
            try:
                # Import the module
                module = importlib.import_module(module_full_name)
                # Get all classes from the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if the class inherits from cls but is not cls itself
                    if issubclass(obj, cls) and obj != cls:
                        if not obj.__subclasses__():
                            subclasses[module_name] = obj
            except ImportError as e:
                logger.error(f"Failed to import module {module_full_name}: {e}")
                continue

        return subclasses
