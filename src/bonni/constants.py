import jax
from typing import Callable, Union

import numpy as np


INPUT_FN_TYPE = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

