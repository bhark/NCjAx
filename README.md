# PyNCA

Neural Cellular Automata as a computing substrate, implemented in JAX

> [!NOTE]
> This is a WIP. It is, at the moment, completely undocumented and not very robust. 

### Roadmap

- [x] Fundamental NCA substrate
- [x] I/O tooling (sending data to, and receiving data from, the substrate)
- [x] Trainable convolutional filters to replace/extend identity+laplacian
- [x] "Fire rate" (stochastic per-cell dropout) as a stability measure
- [x] Pretraining helper (simple identity mapping to help escape local minima)
- [x] Trainable gain
- [x] Simple API interface
- [ ] Pool sampling for robustness
- [ ] Stochastic damage to cells during processing
- [ ] (Maybe) provide a training routine (probably DQN)
- [ ] Solving CartPole!
- [ ] Some sort of documentation

## The papers

This implementation takes a bit from each of the following very nice papers:

- (1): [A Path to Universal Neural Cellular Automata](https://arxiv.org/pdf/2505.13058)
- (2): [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- (3): [Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent](https://arxiv.org/abs/2106.15240)