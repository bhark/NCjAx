# NCjAx

Neural Cellular Automata as a universal computing substrate accelerated with JAX.

> [!NOTE]
> This is a WIP. It is, at the moment, undocumented and suffers from bad manners (unstable, unreliable, difficult to train).

## Usage

Following is a basic introduction to the NCjAx API. This should be enough to play around and get a feel for the substrate. For more advanced usage, import directly from core files.

### Initialization

```
from NCjAx import NCA, Config

# set up a config
# the only required fields are num_input_nodes and num_output_nodes
conf = Config(
    num_input_nodes=num_input,
    num_output_nodes=num_output,
    k_default=65,
    grid_size=16,
    hidden_channels=3,
    perception='learned3x3',
    hidden=30,
)

# nice little config wrapper - avoids threading config every time, and provides some other quality-of-life
nca = NCA(conf)

key, init_key, pretrain_key = jax.random.split(key, 3)

# initialize parameters
params = nca.init_params(init_key)

# curriculum pretraining - helps to break out of local minima from the get-go
params, key, pretrain_accuracy = nca.pretrain(params, pretrain_key, steps=3000)
```

### Processing

The `process()` helper is just a shortcut to help you forward the substrate. It sends input, processes `K` ticks, and reads output. 


```
key, processing_key = jax.random.split(key)
output, next_state = nca.process(nca_state, nca_params, processing_key, input)
```

Note that although NCjAx is functionally pure, NCA is by nature stateful. Preserve the state and pass it back the next time around.


## Roadmap

If you feel like contributing, be my guest. Here's what's missing to align properly with the papers (and hopefully achieve some sort of learning ability beyond input->output mapping):

- [x] Fundamental NCA substrate
- [x] I/O tooling (sending data to, and receiving data from, the substrate)
- [x] Trainable convolutional filters to replace/extend identity+laplacian
- [x] "Fire rate" (stochastic per-cell dropout) as a stability measure
- [x] Pretraining helper (simple identity mapping to help escape local minima)
- [x] Trainable gain
- [x] Simple API interface
- [ ] Solving CartPole!

So... Solving cartpole. If anyone manages, please let me know. At the moment, I haven't figured out how to break out of a degenerate local minima where the same action is called constantly. 

## The papers

This implementation takes a bit from each of the following very nice papers:

- (1): [A Path to Universal Neural Cellular Automata](https://arxiv.org/pdf/2505.13058)
- (2): [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- (3): [Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent](https://arxiv.org/abs/2106.15240)