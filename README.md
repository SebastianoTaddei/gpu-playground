# GPU Playground

Various tests for GPU compute using multiple backends, with a focus on linear
algebra.

## Roadmap

The idea is to play around with various linear algebra concepts for the GPU.
For now the focus is on:

- [ ] Vector-vector summation.
- [ ] Vector-vector multiplication.
- [ ] Vector-vector multiplication (element-wise).
- [ ] Matrix-vector summation.
- [ ] Matrix-vector multiplication.
- [ ] Matrix-vector multiplication (element-wise).
- [ ] Matrix-matrix summation.
- [ ] Matrix-matrix multiplication.
- [ ] Matrix-matrix multiplication (element-wise).
- [ ] Solving linear systems (*i.e.,* $Ax = b$)

As such we will create several shaders/kernels to compute this operations in an
efficient manner.

Additionally, we would like to investigate how to support several compute
backends. For now the focus is on:

- [-] Metal.
- [ ] CUDA.

## Contributing

All contributions are welcome, take this as a playground to experiment with GPU
programming!

## License

See [LICENSE](./LICENSE).
