# pytorch-wgan

Core concepts:
1. Kantorivich-Rubinstein theorem applied to Wasserstein distance
2. WGAN training procedure with parametrized generator and critic
   - Weight clipping
   - Gradient penalty
3. Instantiations of WGAN
   - MLP
   - DCGAN
     - Architecture guidelines
     - Batch normalization
     - ConvTranspose

## Related work

We provide an (possibly) alternative result to the following work:

- https://github.com/kremerj/gan: WGAN on 1d distribution; however, it uses one-sided gradient penalty and diverges after good initial learning
- https://chunliangli.github.io/docs/dltp17gan.pdf: shows that WGAN cannot learn simple 1d distributions

## WGAN-GP with MLPs on 1D distributions (GIFs)

LayerNorm and two-sided gradient penalty were required for this to work nicely.

### Normal

<img src="gifs/normal.gif" alt="normal" width="500">

### Uniform

<img src="gifs/uniform.gif" alt="normal" width="500">

### Bimodal

<img src="gifs/bimodal.gif" alt="normal" width="500">
