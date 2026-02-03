"""
Device Detection and Selection for PyTorch

This module provides automatic device detection and selection for PyTorch,
supporting CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU fallback.

The module follows a priority order: CUDA > MPS > CPU, selecting the best
available accelerator for your hardware.

Usage:
    from device import get_device, print_device_info

    device = get_device()
    model = MyModel().to(device)

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import torch


def get_device(verbose: bool = False) -> str:
    """
    Detect and return the best available PyTorch device.

    Checks for available hardware accelerators in priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. CPU (fallback)

    Args:
        verbose: If True, print device information during detection.
            Default is False.

    Returns:
        A device string suitable for use with tensor.to() or model.to().
        One of: "cuda", "mps", or "cpu".

    Example:
        >>> device = get_device()
        >>> print(device)
        'mps'
        >>> tensor = torch.randn(3, 3).to(device)
        >>> model = MyModel().to(device)

    Note:
        For MPS, both availability and build status are checked to ensure
        the backend is fully functional.
    """
    # Check if CUDA (NVIDIA GPU) is available on this system
    use_cuda = torch.cuda.is_available()
    # Check for MPS (Apple Silicon) availability and build status
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    # Select device based on priority: CUDA > MPS > CPU
    if use_cuda:
        # Use CUDA if available (highest priority)
        device = "cuda"
    elif use_mps:
        # Use MPS if CUDA not available but MPS is (second priority)
        device = "mps"
    else:
        # Fall back to CPU if no accelerators available
        device = "cpu"
    # Optionally print device information if verbose mode is enabled
    if verbose:
        # Call print_device_info to display device details
        print_device_info()
    # Return the selected device string
    return device


def print_device_info() -> None:
    """
    Print detailed information about available PyTorch devices.

    Displays the PyTorch version and availability status of all supported
    accelerators (CUDA, MPS). For CUDA devices, also prints the GPU name.

    This function is useful for debugging hardware detection issues or
    verifying that the expected accelerator is available.

    Example:
        >>> print_device_info()
        Torch version: 2.1.0
        CUDA available: False
        MPS available: True
        MPS built: True
        Using device: mps

    Note:
        This function prints to stdout and returns None.
    """
    # Print PyTorch version for debugging and verification
    print("Torch version:", torch.__version__)
    # Check for CUDA (NVIDIA GPU) availability
    use_cuda = torch.cuda.is_available()
    # Check for MPS (Apple Silicon) availability and build status
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    # Check if CUDA is available and print GPU information
    if use_cuda:
        # Get the name of the first CUDA GPU device
        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Unknown CUDA GPU"
        )
        # Print CUDA availability status
        print("CUDA available:", True)
        # Print the GPU device name
        print("GPU name:", gpu_name)
    else:
        # Print that CUDA is not available
        print("CUDA available:", False)
    # Print MPS availability status
    print("MPS available:", torch.backends.mps.is_available())
    # Print whether MPS backend was built
    print("MPS built:", torch.backends.mps.is_built())
    # Determine the selected device based on availability priority
    if use_cuda:
        # Use CUDA if available (highest priority)
        device = "cuda"
    elif use_mps:
        # Use MPS if CUDA not available but MPS is (second priority)
        device = "mps"
    else:
        # Fall back to CPU if no accelerators available
        device = "cpu"
    # Print the selected device
    print("Using device:", device)


def get_device_info() -> dict:
    """
    Get device information as a dictionary.

    Returns a dictionary containing detailed information about the PyTorch
    environment and available accelerators. Useful for logging or programmatic
    access to device information.

    Returns:
        A dictionary with the following keys:
            - torch_version: PyTorch version string.
            - cuda_available: Boolean indicating CUDA availability.
            - cuda_device_name: GPU name if CUDA available, else None.
            - mps_available: Boolean indicating MPS availability.
            - mps_built: Boolean indicating if MPS backend was built.
            - device: The selected device string ("cuda", "mps", or "cpu").

    Example:
        >>> info = get_device_info()
        >>> print(info["device"])
        'mps'
        >>> print(info["torch_version"])
        '2.1.0'

    Note:
        This function does not print anything; use print_device_info() for
        console output.
    """
    # Check for CUDA (NVIDIA GPU) availability for dictionary
    use_cuda = torch.cuda.is_available()
    # Check for MPS (Apple Silicon) availability and build status
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    # Initialize CUDA device name as None
    cuda_device_name = None
    # Check if CUDA is available to get device name
    if use_cuda:
        # Get the name of the first CUDA GPU
        cuda_device_name = torch.cuda.get_device_name(0)
    # Determine selected device based on availability priority
    if use_cuda:
        # Use CUDA if available (highest priority)
        device = "cuda"
    elif use_mps:
        # Use MPS if CUDA not available but MPS is (second priority)
        device = "mps"
    else:
        # Fall back to CPU if no accelerators available
        device = "cpu"
    # Build and return the device information dictionary
    return {
        "torch_version": torch.__version__,
        "cuda_available": use_cuda,
        "cuda_device_name": cuda_device_name,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "device": device,
    }
