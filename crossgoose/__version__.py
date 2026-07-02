"""Package version for crossgoose."""
import importlib.metadata

try:
    __version__ = importlib.metadata.version("crossgoose")
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.3.5"  # fallback for development
