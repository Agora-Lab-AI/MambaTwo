
"""
Mamba-2 Implementation in PyTorch

This module implements the Mamba-2 architecture, a state-space model based
transformer alternative. It includes robust device handling, type checking,
and comprehensive logging.

Features:
- Multi-device support (CPU/GPU/Multi-GPU)
- Robust error handling and validation
- Comprehensive shape tracking and logging
- Type hints and documentation
"""

import math
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor


class DeviceType(Enum):
    """Supported device types for model execution."""

    CPU = auto()
    GPU = auto()
    MULTI_GPU = auto()


@dataclass
class Mamba2Config:
    """Configuration for Mamba-2 model.

    Args:
        d_model: Model dimension
        depth: Number of Mamba blocks
        d_state: State dimension for SSM
        d_conv: Convolution kernel size
        expand_factor: Expansion factor for inner dimension
        device_type: Type of device to run on
        dtype: Data type for model parameters
        distributed: Whether to use distributed training
    """

    d_model: int
    depth: int
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    device_type: DeviceType = DeviceType.CPU
    dtype: torch.dtype = torch.float32
    distributed: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {self.d_model}"
            )
        if self.depth <= 0:
            raise ValueError(
                f"depth must be positive, got {self.depth}"
            )
        if self.d_state <= 0:
            raise ValueError(
                f"d_state must be positive, got {self.d_state}"
            )
        if self.d_conv <= 0:
            raise ValueError(
                f"d_conv must be positive, got {self.d_conv}"
            )
        if self.expand_factor <= 0:
            raise ValueError(
                f"expand_factor must be positive, got {self.expand_factor}"
            )


class DeviceManager:
    """Manages device placement and data movement for the model."""

    def __init__(self, config: Mamba2Config):
        self.config = config
        self.device = self._setup_device()

    def _setup_device(self) -> torch.device:
        """Set up the appropriate device based on configuration."""
        if self.config.device_type == DeviceType.CPU:
            return torch.device("cpu")

        if not torch.cuda.is_available():
            warnings.warn(
                "GPU requested but CUDA is not available. Falling back to CPU."
            )
            return torch.device("cpu")

        if self.config.device_type == DeviceType.MULTI_GPU:
            if torch.cuda.device_count() < 2:
                warnings.warn(
                    "Multi-GPU requested but less than 2 GPUs available. Using single GPU."
                )
                return torch.device("cuda:0")
            return torch.device("cuda")

        return torch.device("cuda:0")

    def to_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to appropriate device with error handling."""
        try:
            return tensor.to(self.device, dtype=self.config.dtype)
        except RuntimeError as e:
            logger.error(
                f"Failed to move tensor to device {self.device}: {e}"
            )
            raise


class SSM(nn.Module):
    """Structured State Space Model component of Mamba-2.

    Implements the core state space transformation with selective scan.

    Args:
        d_model: Model dimension
        d_state: State dimension
        dt_rank: Rank of Δ projection
        device_manager: Device management instance
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dt_rank: int,
        device_manager: DeviceManager,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.device_manager = device_manager

        # Initialize parameters
        self.A = nn.Parameter(
            torch.randn(d_state, d_state) / math.sqrt(d_state)
        )
        self.D = nn.Parameter(
            torch.randn(d_model) / math.sqrt(d_model)
        )
        self.dt_projs = nn.Parameter(
            torch.randn(dt_rank, d_model) / math.sqrt(dt_rank)
        )

    def forward(self, x: Tensor, B: Tensor, C: Tensor) -> Tensor:
        """
        Forward pass of SSM.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            B: B matrix (batch, seq_len, d_inner, d_state)
            C: C matrix (batch, seq_len, d_inner, d_state)

        Returns:
            Tensor: Output tensor (batch, seq_len, d_model)

        Raises:
            RuntimeError: If tensor dimensions don't match expected shapes
        """
        self._validate_input_shapes(x, B, C)

        batch, seq_len, d_model = x.shape

        # Compute Δ
        torch.einsum("rd,bsd->bsr", self.dt_projs, x)

        # Discretize A
        A_expanded = self.A.unsqueeze(0).unsqueeze(0)
        A_expanded = A_expanded.expand(batch, seq_len, -1, -1)
        dA = torch.exp(A_expanded)

        # Initialize state
        h = self.device_manager.to_device(
            torch.zeros(batch, self.d_state)
        )
        y = []

        # Selective scan with error checking
        try:
            for t in range(seq_len):
                h = torch.bmm(dA[:, t], h.unsqueeze(-1)).squeeze(-1)
                h = h + torch.einsum("bmd,bm->bd", B[:, t], x[:, t])
                y_t = torch.einsum("bd,bmd->bm", h, C[:, t])
                y.append(y_t)
        except RuntimeError as e:
            logger.error(
                f"Error in selective scan at position {t}: {e}"
            )
            raise

        y = torch.stack(y, dim=1)
        return y + self.D.unsqueeze(0).unsqueeze(0) * x

    def _validate_input_shapes(self, x: Tensor, B: Tensor, C: Tensor):
        """Validate input tensor shapes."""
        if x.dim() != 3:
            raise ValueError(
                f"Expected x to have 3 dimensions, got {x.dim()}"
            )
        if B.dim() != 4:
            raise ValueError(
                f"Expected B to have 4 dimensions, got {B.dim()}"
            )
        if C.dim() != 4:
            raise ValueError(
                f"Expected C to have 4 dimensions, got {C.dim()}"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected x last dim to be {self.d_model}, got {x.size(-1)}"
            )


class Mamba2Block(nn.Module):
    """Single block of Mamba-2 architecture.

    Args:
        config: Model configuration
        device_manager: Device management instance
    """

    def __init__(
        self, config: Mamba2Config, device_manager: DeviceManager
    ):
        super().__init__()
        self.config = config
        self.device_manager = device_manager
        self.d_inner = config.d_model * config.expand_factor

        # Projections
        self.in_proj_x = nn.Linear(config.d_model, self.d_inner)
        self.in_proj_b = nn.Linear(
            config.d_model, self.d_inner * config.d_state
        )
        self.in_proj_c = nn.Linear(
            config.d_model, self.d_inner * config.d_state
        )

        # Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding="same",
            groups=self.d_inner,
        )

        # SSM
        self.ssm = SSM(
            d_model=self.d_inner,
            d_state=config.d_state,
            dt_rank=8,
            device_manager=device_manager,
        )

        self.norm = nn.GroupNorm(
            num_groups=1, num_channels=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Mamba-2 block."""
        batch, seq_len, _ = x.shape

        # Projections
        x_projected = self.in_proj_x(x)
        b_projected = self.in_proj_b(x)
        c_projected = self.in_proj_c(x)

        # Reshape with dimension checking
        try:
            B = b_projected.view(
                batch, seq_len, self.d_inner, self.config.d_state
            )
            C = c_projected.view(
                batch, seq_len, self.d_inner, self.config.d_state
            )
        except RuntimeError as e:
            logger.error(f"Failed to reshape projections: {e}")
            raise

        # Process
        x_conv = self.conv1d(x_projected.transpose(-1, -2)).transpose(
            -1, -2
        )
        x_ssm = self.ssm(x_conv, B, C)
        x_norm = self.norm(x_ssm.transpose(-1, -2)).transpose(-1, -2)

        return self.out_proj(x_norm)


class Mamba2(nn.Module):
    """Complete Mamba-2 architecture."""

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.device_manager = DeviceManager(config)

        # Create blocks
        self.blocks = nn.ModuleList(
            [
                Mamba2Block(config, self.device_manager)
                for _ in range(config.depth)
            ]
        )

        # Move model to appropriate device
        self.to(self.device_manager.device)

        if config.distributed:
            self.blocks = nn.DataParallel(self.blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of complete Mamba-2 model."""
        x = self.device_manager.to_device(x)

        for block in self.blocks:
            try:
                x = x + block(x)
            except RuntimeError as e:
                logger.error(f"Error in block forward pass: {e}")
                raise

        return x


def create_mamba2_model(
    config: Mamba2Config, seed: Optional[int] = None
) -> Mamba2:
    """
    Create a Mamba-2 model with specified configuration.

    Args:
        config: Model configuration
        seed: Random seed for reproducibility

    Returns:
        Configured Mamba-2 model

    Raises:
        RuntimeError: If model creation fails
    """
    if seed is not None:
        torch.manual_seed(seed)

    try:
        model = Mamba2(config)
        logger.info(
            f"Created Mamba-2 model: d_model={config.d_model}, "
            f"depth={config.depth}, device={model.device_manager.device}"
        )
        return model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise


def example_usage():
    """Example usage of Mamba-2 model."""
    # Configure logging
    logger.add("mamba2.log", rotation="500 MB")

    # Create configuration
    config = Mamba2Config(
        d_model=256,
        depth=4,
        device_type=(
            DeviceType.GPU
            if torch.cuda.is_available()
            else DeviceType.CPU
        ),
        distributed=torch.cuda.device_count() > 1,
    )

    # Create model
    model = create_mamba2_model(config, seed=42)

    # Example forward pass
    batch_size, seq_len = 32, 128
    x = torch.randn(batch_size, seq_len, config.d_model)

    logger.info("Starting forward pass")
    with torch.no_grad():
        output = model(x)
    logger.info(
        f"Forward pass complete. Output shape: {output.shape}"
    )


if __name__ == "__main__":
    example_usage()
