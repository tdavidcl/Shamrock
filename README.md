<img src="doc/mkdocs/docs/assets/no_background_nocolor.png"  width="600">

![badge1](https://github.com/Shamrock-code/Shamrock/actions/workflows/on_push_main.yml/badge.svg?branch=main) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# The Shamrock code

Shamrock is a general purpose HPC hydrodynamics simulation code focused on astrophysical contexts.
The whole code is **C++17** by default, and all accelerated parts are done using **SYCL**,
which can be directly compiled directly to native **CUDA**, **ROCM**, **OpenMP** and much more.
Shamrock also supports multiple GPUs and Heterogeneous clusters using **MPI**.
This code aims to be:
- Modern
- Modular
- Fast
- Portable

## Getting in touch

Join us on [Discord](https://discord.gg/Q69s5buyr5), or [Slack](https://shamrock-code.slack.com)! Alternatively, open a discussion or issue in this repository.

## Contributing

Shamrock accept contributions through github pull request :
1. Code contributions via [Pull request](https://github.com/Shamrock-code/Shamrock/compare)
2. Documentation contributions via [Pull request](https://github.com/Shamrock-code/Shamrock/compare)
3. Issue report & feature requests via [Github issues](https://github.com/Shamrock-code/Shamrock/issues/new/choose)

If you want to contribute please fork the code and submit your pull requests from your fork.

## Citing the code

[![bibtex](https://img.shields.io/badge/bibtex-file-green.svg)](./CITATION.bib)
[![Nasa ads](https://img.shields.io/badge/nasa_ADS-2503.09713-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713)
[![arXiv](https://img.shields.io/badge/arXiv-2503.09713-b31b1b.svg)](https://arxiv.org/abs/2503.09713)

## Compiler support

Compiler config | Support
---|---
DPC++ CUDA | ![badge2](https://badgen.net/static/DPC++%2FCUDA/yes/green)
DPC++ ROCM | ![badge2](https://badgen.net/static/DPC++%2FHIP:ROCM/yes/green)
AdaptiveCPP OpenMP | ![badge2](https://badgen.net/static/ACPP%2FOpenMP/yes/green)
AdaptiveCPP ROCM | ![badge2](https://badgen.net/static/ACPP%2FROCM/yes/green)
AdaptiveCPP CUDA | ![badge2](https://badgen.net/static/ACPP%2FCUDA/yes/green)
AdaptiveCPP SSCP | ![badge2](https://badgen.net/static/ACPP%2FSSCP/yes/green)


# Documentation

We provide both a book like documentation and the more classic doxygen style for more details about the sources
 - The documentation is available here: [Shamrock documentation](https://shamrock-code.github.io/Shamrock/mkdocs/index.html)

# Getting started

The whole getting started guide can be found here: [Getting started](https://shamrock-code.github.io/Shamrock/mkdocs/usermanual/quickstart/)

Note that a convenient way to pull the Shamrock repo is the following command to also pull the submodules:
```bash
git clone --recurse-submodules https://github.com/Shamrock-code/Shamrock.git
```

# Licensing

Shamrock is licensed under the `CeCILL Free Software License Agreement v2.1`. See [License (french version)](./LICENSE) or [License (english version)](./LICENSE.en) for more details.

Copyright 2021-2024 Timothée David--Cléris

SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
