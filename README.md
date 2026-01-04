# GPU Playground

Various tests for GPU compute using multiple backends, with a focus on linear
algebra.

## Roadmap

The idea is to play around with various linear algebra concepts for the GPU.
For now the focus is on:

- [x] Vector-vector summation.
- [x] Vector-vector multiplication.
- [ ] Vector-vector multiplication (element-wise).
- [ ] Matrix-vector summation.
- [x] Matrix-vector multiplication.
- [ ] Matrix-vector multiplication (element-wise).
- [x] Matrix-matrix summation.
- [x] Matrix-matrix multiplication.
- [ ] Matrix-matrix multiplication (element-wise).
- [ ] Solving linear systems (*i.e.,* $Ax = b$)

As such we will create several shaders/kernels to compute this operations in an
efficient manner.

Additionally, we would like to investigate how to support several compute
backends. For now the focus is on:

- [x] Serial (serial backend on CPU).
- [x] Eigen (vectorised backend on CPU, thanks to the
[Eigen](https://libeigen.gitlab.io) library).
- [x] SIMD (vectorised backend on CPU, thanks to the
[xsimd](https://xsimd.readthedocs.io/en/latest/) library).
- [x] Metal (GPU backend).
- [ ] CUDA (GPU backend).

## Performance

The focus of this library is on learning how vectorised backends work, rather
than on performance. That said, we will try to optimise it to the best we can.

## Contributing

All contributions are welcome, take this as a playground to experiment with GPU
programming!

## License

See [LICENSE](./LICENSE).
