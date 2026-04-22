"""Model persistence: saving, loading, and a name-to-class registry."""

from diffsurrogate.persistence.loader import load_model
from diffsurrogate.persistence.registry import MODEL_REGISTRY, class_for
from diffsurrogate.persistence.saver import save_model

__all__ = ["save_model", "load_model", "MODEL_REGISTRY", "class_for"]
