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

    - 1 information channel (immutable at input cells)
    - H hidden channels
    - 2 identifier channels: input_flag, output_flag (immutable)
    """

    grid_size: int = 16
    hidden_channels: int = 8
    num_input_nodes: int = 8
    num_output_nodes: int = 2

    perception: PerceptionMode = 'id_lap'
    hidden: int = 64 # mlp hidden width

    fire_rate: float = 1.0 # per-cell update prob in [0, 1]
    k_default: int = 1 # number of updates/ticks per step

    dtype: DTypeLike = field(default=jnp.float32, repr=False)

    padding: PaddingMode = 'reflect'

    # -- properties --

    @property
    def info_channels(self) -> int:
        return 1

    @property
    def id_channels(self) -> int:
        return 2 # input flag, output flag

    @property
    def C(self) -> int:
        ''' total number of channels '''
        return self.info_channels + self.hidden_channels + self.id_channels

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

    # fixed channel indices
    @property
    def idx_info(self) -> int:
        return 0

    @property
    def idx_in_flag(self) -> int:
        return 1 + self.hidden_channels

    @property
    def idx_out_flag(self) -> int:
        return 1 + self.hidden_channels + 1

    # -- validation --
    def __post_init__(self):
        if not (self.grid_size > 0):
            raise ValueError(f'Invalid grid size: {self.grid_size}')
        if self.hidden <= 0:
            raise ValueError(f'Hidden width must be larger than 0, got {self.hidden}')
        if not (0.0 <= self.fire_rate <= 1.0):
            raise ValueError(f'Fire rate must be in [0,1], got {self.fire_rate}')
        if self.k_default <= 0:
            raise ValueError(f'K must be larger than 0, got {self.k_default}')
        if self.perception not in ('id_lap', 'raw9'):
            raise ValueError(f'Unsupported perception mode: {self.perception}')
        if self.padding != 'reflect':
            raise ValueError(f'Unsupported padding mode: {self.padding}')
        if self.hidden_channels < 0:
            raise ValueError('hidden_channels must be >= 0')
        if self.num_input_nodes <= 0:
            raise ValueError('num_input_nodes must be >= 1')
        if self.num_output_nodes <= 0:
            raise ValueError('num_output_nodes must be >= 1')
        object.__setattr__(self, 'dtype', jnp.dtype(self.dtype))

    
    # -- convenience stuff --
    def describe(self) -> str:
        return(
            'NCA config:\n'
            f' Grid: C={self.C} (info=1, hidden={self.hidden_channels}, id=2), size={self.grid_size}\n'
            f' Perception: {self.perception} (F={self.F})\n'
            f' MLP: hidden={self.hidden}\n'
            f' Dynamics: fire_rate={self.fire_rate}, K={self.k_default}\n'
            f' I/O nodes: inputs={self.num_input_nodes}, outputs={self.num_output_nodes}\n'
            f' Numerics: dtype={self.dtype.name}, padding={self.padding}\n'
            f' Derived: input_feats_per_cell={self.input_feats_per_cell}'
        )