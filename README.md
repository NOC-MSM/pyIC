# pyIC

pyIC is a python package to generate initial conditions for regional NEMO model configurations.

## Quick Start

Detailed instructions for getting started with using this package are available in the [documentation](https://noc-msm.github.io/pyIC/#quick-start).

```python
>>> import pyic

```

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.11:

```
conda create -n DEVELOP -c conda-forge python=3.11
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Sync with the latest [template](https://github.com/ecmwf-projects/cookiecutter-conda-package) (optional): `make template-update`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`
1. The documentation builds automatically on push (the default version of this cookie-cutter package uses Sphinx, which we're not using here).

## License

```
Copyright 2024, NOC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
