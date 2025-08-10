from dataclasses import dataclass, field
from typing import Literal
import jax.numpy as jnp
from jax.typing import DTypeLike

PerceptionMode = Literal['id_lap', 'raw9']
PaddingMode = Literal['reflect']

@dataclass(slots=True, frozen=True)
class Config:
    """
    config for NCA instance

    perception modes:
    - 'id_lap' -> identity and laplacian
    - 'raw9' -> raw values from self and neighbours

    padding modes:
    - 'reflect' -> mirrors grid at the border
    """

    io_channels: int # I/O state channels
    grid_size: int = 16
    hidden_channels: int = 8 # hidden state channels

    perception: PerceptionMode = 'id_lap'
    hidden: int = 64 # mlp hidden width

    fire_rate: float = 1.0 # per-cell update prob in [0, 1]
    k_default: int = 1 # number of updates/ticks per step

    dtype: DTypeLike = field(default=jnp.float32, repr=False)

    padding: PaddingMode = 'reflect'

    # -- properties --

    @property
    def C(self) -> int:
        ''' total number of channels '''
        return self.io_channels + self.hidden_channels

    @property
    def F(self) -> int:
        ''' number of perception features per channel '''
        return 2 if self.perception == 'id_lap' else 9

    @property
    def input_feats_per_cell(self) -> int:
        ''' size of per-cell perception vector fed to mlp '''
        return self.C * self.F

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        ''' canonical (C, H, W) shape tuple '''
        return (self.C, self.grid_size, self.grid_size)

    # -- validation --
    def __post_init__(self):
        # basic integer checks
        if not (self.C > 0):
            raise ValueError(f'Invalid channel count: {self.C}')

        if not (self.grid_size > 0):
            raise ValueError(f'Invalid grid size: {self.grid_size}')

        # hidden width
        if self.hidden <= 0:
            raise ValueError(f'Hidden width must be larger than 0, got {self.hidden}')

        # fire rate
        if not (0.0 <= self.fire_rate <= 1.0):
            raise ValueError(f'Fire rate must be in [0,1], got {self.fire_rate}')

        # K default
        if self.k_default <= 0:
            raise ValueError(f'K must be larger than 0, got {self.k_default}')

        # perception
        if self.perception not in ('id_lap', 'raw9'):
            raise ValueError(f'Unsupported perception mode: {self.perception}')

        # padding
        if self.padding != 'reflect':
            raise ValueError(f'Unsupported padding mode: {self.padding}')

        object.__setattr__(self, 'dtype', jnp.dtype(self.dtype))

    
    # -- convenience stuff --
    def describe(self) -> str:
        return(
            'NCA config:\n'
            f' Grid: C={self.C}, size={self.grid_size}\n'
            f' Perception: {self.perception} (F={self.F})\n'
            f' MLP: hidden={self.hidden}\n'
            f' Dynamics: fire_rate={self.fire_rate}, K={self.k_default}\n'
            f' Numerics: dtype={self.dtype.name}, padding={self.padding}\n'
            f' Derived: input_feats_per_cell={self.input_feats_per_cell}'
        )