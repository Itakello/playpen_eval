import importlib
import inspect
from abc import ABC
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseClass(ABC):
    @classmethod
    def get_others(cls) -> dict[str, type]:
        """Get all non-base classes in the same directory that inherit from this class."""
        # Get the directory of the current file
        calling_frame = inspect.stack()[1]
        calling_module = inspect.getmodule(calling_frame[0])

        # Handle case where module path cannot be determined
        module_file = getattr(calling_module, "__file__", None)
        if not module_file:
            return {}

        current_dir = Path(str(module_file)).parent

        # Get all .py files in the directory
        py_files = [
            f
            for f in current_dir.glob("*.py")
            if f.is_file()
            and f.name != "__init__.py"
            and not f.name.startswith("base_")
        ]

        subclasses = {}
        for file in py_files:
            # Convert file path to module path
            module_full_name = f"{cls.__module__.rsplit('.', 1)[0]}.{file.stem}"
            module_name = file.stem.split("_")[0]
            try:
                # Import the module
                module = importlib.import_module(module_full_name)
                # Get all classes from the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if the class inherits from cls but is not cls itself
                    if issubclass(obj, cls) and obj != cls:
                        subclasses[module_name] = obj
            except ImportError:
                continue

        return subclasses
