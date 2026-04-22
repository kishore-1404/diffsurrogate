"""diffsurrogate — surrogate modeling for diffractive scattering observables.

Six surrogate paradigms evaluated head-to-head against QCD lookup tables:
neural nets, exact & sparse GPs, PINNs, FNOs, deep GPs, and polynomial chaos.
"""

__version__ = "0.1.0"

from diffsurrogate.config import Config, load_config
from diffsurrogate.models.base import SurrogateModel

__all__ = ["Config", "load_config", "SurrogateModel", "__version__"]
