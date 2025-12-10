"""
GPU configuration utilities for TensorFlow.
"""

import tensorflow as tf


def configure_gpu():
    """
    Configure GPU memory growth to avoid allocating all GPU memory at once.
    This allows multiple processes to share the GPU and prevents OOM errors.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured {len(gpus)} GPU(s) with memory growth enabled")
            
        except RuntimeError as e:
            print(f"✗ GPU configuration error: {e}")
    else:
        print("⚠ No GPUs detected. Training will use CPU.")


def list_gpus():
    """
    List available GPUs and their details.
    
    Returns:
        List of available GPU devices
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\nAvailable GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("No GPUs available")
    
    return gpus


def set_gpu_memory_limit(memory_limit_mb: int, gpu_index: int = 0):
    """
    Set a hard memory limit on a specific GPU.
    
    Args:
        memory_limit_mb: Memory limit in megabytes
        gpu_index: Index of the GPU to configure (default: 0)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus and gpu_index < len(gpus):
        try:
            tf.config.set_logical_device_configuration(
                gpus[gpu_index],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            print(f"✓ Set memory limit of {memory_limit_mb} MB on GPU {gpu_index}")
        except RuntimeError as e:
            print(f"✗ Error setting memory limit: {e}")
    else:
        print(f"✗ GPU {gpu_index} not found")


def disable_gpu():
    """
    Disable GPU usage and force TensorFlow to use CPU only.
    Useful for debugging or when GPU is unavailable.
    """
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled. Using CPU only.")
    except RuntimeError as e:
        print(f"✗ Error disabling GPU: {e}")