import importlib
import inspect
import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

from itakello_logging import ItakelloLogging

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class BaseClass(ABC):
    @classmethod
    def _get_module_path(cls) -> tuple[Path | None, str]:
        """Helper method to get the module path and base module name."""
        calling_frame = inspect.stack()[1]
        calling_module = inspect.getmodule(calling_frame[0])

        # Handle case where module path cannot be determined
        module_file = getattr(calling_module, "__file__", None)
        if not module_file:
            return None, ""

        current_dir = Path(str(module_file)).parent
        base_module_name = cls.__module__.rsplit(".", 1)[0]
        return current_dir, base_module_name

    @classmethod
    def _get_py_files(cls, current_dir: Path) -> list[Path]:
        """Helper method to get relevant Python files in the directory."""
        return [
            f
            for f in current_dir.glob("*.py")
            if f.is_file()
            and f.name != "__init__.py"
            and not f.name.startswith("base_")
        ]

    @classmethod
    def get_all_subclasses(cls) -> dict[str, type]:
        """Get all subclasses in the same directory that inherit from this class."""
        current_dir, base_module_name = cls._get_module_path()
        if not current_dir:
            return {}

        py_files = cls._get_py_files(current_dir)
        return cls._load_subclasses(py_files, base_module_name)

    @classmethod
    def get_specific_subclass(cls, subclass_name: str) -> tuple[str, type]:
        """Get a specific subclass by its name.

        Args:
            subclass_name: The name of the subclass to retrieve (case-insensitive)

        Returns:
            A tuple containing the name and the subclass if found, None otherwise
        """
        current_dir, base_module_name = cls._get_module_path()
        if not current_dir:
            return None

        py_files = cls._get_py_files(current_dir)
        subclasses = cls._load_subclasses(py_files, base_module_name)

        # Try to match the subclass name (case-insensitive)
        subclass_name = subclass_name.lower()
        for name, subclass in subclasses.items():
            if name.lower() == subclass_name:
                return subclass

        logger.warning(f"Subclass '{subclass_name}' not found")
        return None

    @classmethod
    def _load_subclasses(
        cls, py_files: list[Path], base_module_name: str
    ) -> dict[str, type]:
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

    def load_credentials(self, backend: str) -> str:
        """Load API key from environment variables."""
        env_var_name = f"{backend.upper()}_API_KEY"
        key = os.getenv(env_var_name)
        if key is None:
            raise ValueError(f"API key for {backend} not found.")
        return key
