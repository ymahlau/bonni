from typing import Callable

import numpy as np


INPUT_FN_TYPE = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
