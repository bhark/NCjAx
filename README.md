# Neural Cellular Automata as computing substrate

> [!NOTE]
> This is a WIP. It is, at the moment, completely undocumented and not very robust. 

This library attempts to use NCA as a trainable policy. It is intentionally lean - bring your own training routine. 

### Current state

I have yet to be able to reproduce the results claimed by Alexandre Variengien et. al. in (3). They use a lot of very clever tricks to maintain stability which are still lacking from this implementation (pretraining on a simpler task, pool sampling, damage). 

### What's what?

This is a playground, not a production library. Don't expect anything to align perfectly with conventions or papers. 

## The papers

This implementation takes a bit from each of the following papers (and leaves some things out).

- (1): [A Path to Universal Neural Cellular Automata](https://arxiv.org/pdf/2505.13058)
- (2): [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- (3): [Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent](https://arxiv.org/abs/2106.15240)