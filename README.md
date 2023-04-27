# Quaru

![quaru_opacity](https://user-images.githubusercontent.com/73998667/228244740-bd136eae-1bc5-4048-a76f-8cb85f4e99c9.png)

![example workflow](https://github.com/DATX11-22/Quaru/actions/workflows/rust.yml/badge.svg)

Repository for group 22 quantum computer simulator bachelor's thesis.

## External Links

- [Google Drive](https://drive.google.com/drive/folders/1SPfrqoUSkliOfUi64RXRsj8BRy-veqLI?usp=sharing)
- [Canvas Page](https://chalmers.instructure.com/groups/158527)
- [Project Report](https://www.overleaf.com/read/tsphshnkpfxy)
- [Project Plan](https://www.overleaf.com/project/63ca8a6b32ea8a38a590acc1)

## Examples

The simulator features a number of examples located under the /examples directory. These are binary crates you can run with cargo.`<br/>`
To run an example, run the following command: `<br/>`
`cargo run --example <filename>`

Here are all available examples: 

* `grovers`
* `openquasm_example`
* `quantum_teleportation`
* `shors`
* `superdense_coding`

Some examples accept parameters. To provide these, use: `<br/>`
`cargo run --example <filename> -- --flag <value>`

Here is a list of all parameters for each example that accepts them:

* Shors

  * Number to be factorized: `--number <value>`
  * How many times to run: `--n-times <value>`
  * If you want to print debugging messages from the algorithm you need to change  an enviroment variable
    * For linux: `RUST_LOG=debug cargo run --example shors`
    * For windows: `$env:RUST_LOG="debug"; cargo run --example shors; $env:RUST_LOG="off"`
* Grovers

  * State to search for: `--target <value>`
  * Run for statistics of different iteration formulas: `--statistics`
* Quantum teleportation

  * here there are 4 paramaters, `a,b,c` and `d`. These correspond to the representation `[a+bi, c+di]` of the qubit to be teleported

For example: `<br/>`
`cargo run --example grovers -- --target 64`

#### BLAS feature

We provide suport for enabling the feature "blas" in the [ndarray](https://crates.io/crates/ndarray) crate.
When enabling this feature you need to add the [blas-src](https://crates.io/crates/blas-src) crate to your dependencies. You also need to choose one of the five supported blas implementations:

* `accelerate`, which is the one in the [Accelerate](https://developer.apple.com/reference/accelerate) framework (macOS only),
* `blis`, which is the one in [BLIS](https://github.com/flame/blis),
* `intel-mkl`, which is the one in [Intel MKL](https://software.intel.com/en-us/mkl),
* `netlib`, which is the reference one by [Netlib](http://www.netlib.org/), and
* `openblas`, which is the one in [OpenBLAS](http://www.openblas.net/).

```
[dependencies]
blas-src = { version = "0.8", features = ["accelerate"] }
blas-src = { version = "0.8", features = ["blis"] }
blas-src = { version = "0.8", features = ["intel-mkl"] }
blas-src = { version = "0.8", features = ["netlib"] }
blas-src = { version = "0.8", features = ["openblas"] }
```

###### OpenBLAS

If you want to use the [OpenBLAS](http://www.openblas.net/) implementation your toml should look like this:

```
blas-src = { version = "0.8", features = ["openblas"], optional = true}
openblas-src = { version = "0.10", features = ["cblas", "system"], optional = true}
```

And your main rust file needs this line at the top:

```rust
extern crate blas_src;
```

For all implementations you need to install software on your local machine.
