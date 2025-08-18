from dataclasses import dataclass, field
from typing import Literal
import jax.numpy as jnp
from jax.typing import DTypeLike

PerceptionMode = Literal['id_lap', 'raw9', 'learned3x3']
PaddingMode = Literal['reflect', 'zeros']

@dataclass(slots=True, frozen=True)
class Config:
    """
    config for NCA instance

    - 1 information channel (immutable at input cells)
    - H hidden channels
    - 2 identifier channels: input_flag, output_flag (immutable)

    perception modes:
    - 'id_lap': concat(center, laplacian) over all channels
    - 'raw9': concat 9 neighbor taps over all channels
    - 'learned3x3': 3x3 conv with K learnable filters
    """

    grid_size: int = 8
    hidden_channels: int = 3
    num_input_nodes: int = 8
    num_output_nodes: int = 2

    perception: PerceptionMode = 'learned3x3'
    hidden: int = 30 # mlp hidden width

    fire_rate: float = 0.8 # per-cell update prob in [0, 1]
    k_default: int = None

    dtype: DTypeLike = field(default=jnp.float32, repr=False)

    padding: PaddingMode = 'zeros'
    conv_features: int = 20

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
        if self.perception == 'id_lap': return 2
        if self.perception == 'raw9': return 9
        return None # for learned3x3

    @property
    def input_feats_per_cell(self) -> int:
        ''' size of per-cell perception vector fed to mlp '''
        return self.conv_features if self.perception == 'learned3x3' else self.C * int(self.F)

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
        if self.k_default and self.k_default <= 0:
            raise ValueError(f'K must be larger than 0, got {self.k_default}')
        if self.perception not in ('id_lap', 'raw9', 'learned3x3'):
            raise ValueError(f'Unsupported perception mode: {self.perception}')
        if self.hidden_channels < 0:
            raise ValueError('hidden_channels must be >= 0')
        if self.num_input_nodes <= 0:
            raise ValueError('num_input_nodes must be >= 1')
        if self.num_output_nodes <= 0:
            raise ValueError('num_output_nodes must be >= 1')
        if self.perception == 'learned3x3' and self.conv_features <= 0:
            raise ValueError('conv_features must be > 0 for learned3x3')
        object.__setattr__(self, 'dtype', jnp.dtype(self.dtype))
        if not self.k_default:
            object.__setattr__(self, 'k_default', int((self.grid_size / self.fire_rate) * 2)) # compute a sane default

    
    # -- convenience stuff --
    def describe(self) -> str:
        perc = (f"learned3x3(K={self.conv_features}, padding={self.padding})"
                if self.perception == 'learned3x3'
                else f"{self.perception} (F={self.F}, padding={self.padding})")

        return(
            'NCA config:\n'
            f' Grid: C={self.C} (info=1, hidden={self.hidden_channels}, id=2), size={self.grid_size}\n'
            f' Perception: {perc}\n'
            f' MLP: hidden={self.hidden}\n'
            f' Dynamics: fire_rate={self.fire_rate}, K={self.k_default}\n'
            f' I/O nodes: inputs={self.num_input_nodes}, outputs={self.num_output_nodes}\n'
            f' Numerics: dtype={self.dtype.name}, padding={self.padding}\n'
            f' Derived: input_feats_per_cell={self.input_feats_per_cell}'
        )