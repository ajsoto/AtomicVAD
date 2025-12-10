"""
Utilities for setting random seeds for reproducibility.
"""

import numpy as np
import tensorflow as tf


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across numpy, TensorFlow, and Python.
    
    Args:
        seed: Random seed value
    """
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Note: For full determinism, you can uncomment the following line,
    # but it may impact performance
    # tf.config.experimental.enable_op_determinism()