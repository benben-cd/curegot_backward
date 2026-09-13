# Stage 6A clean-install acceptance and repository audit

This stage changes documentation and adds an installed-surface validator only.
The existing production implementation and build configuration remain frozen.
Final release freeze requires reviewer acceptance of a newly built wheel.

## Audit of the actual checkout

1. `pyproject.toml` uses `setuptools.build_meta`; `setup.py` explicitly declares
   `packages=["curegot"]`. No package restructuring is needed.
2. Setuptools includes the package's `.py` modules in regular wheels. Editable
   installs expose those modules from the checkout. Root compatibility shims
   are not declared as `py_modules` and are not part of the installed API.
3. `curegot._internal_numpy` uses `Pybind11Extension` with custom CUDA compilation;
   `curegot._internal_torch` uses `TorchCUDAExtension`. `UnifiedBuildExtension`
   dispatches between them. Their qualified names place the resulting `.so`
   files inside `curegot/`. A source build compiles them, rather than simply
   copying ignored in-place `.so` files. `MANIFEST.in` grafts `src` and `figs`;
   setuptools adds package Python files and extension sources to the sdist.
4. Wheel configuration is structurally sufficient **with a working native
   toolchain**. `python -m build` defaults to isolated builds and requires
   PyTorch, pybind11, requests and setuptools; it can download large dependencies.
   It still needs external CUDA/cuDSS development files and a compiler. CCCL
   3.3.3 is downloaded unless an existing include directory is supplied.
   No built wheel/sdist has been validated in Stage 6A yet. This is a static
   packaging conclusion, not a claim of binary portability.
5. The previous README showed native NumPy/PyTorch solvers but no differentiable
   training API. It also did not distinguish optional runtime PyTorch from the
   mandatory isolated-build dependency. The updated README addresses both.
6. `setup.py`, `pyproject.toml`, and `curegot/__init__.py` each define version
   `0.1.0`. They agree; the duplicate definitions are left unchanged. Installed
   `importlib.metadata.version("curegot")` also reports `0.1.0`.
7. Runtime PyTorch remains optional: NumPy loads first, native torch import is
   attempted inside a broad exception handler, and Python API import catches
   `ImportError`. Absence of torch/the torch extension can leave NumPy usable.
   Do not treat successful `import curegot` alone as proof of torch support.
   If torch imports but its native extension fails, `curegot.torch` can still
   refer to the ordinary PyTorch module; the new validator checks the actual
   native namespace and API. Import failures can be hidden by the existing
   broad handler. This behavior was not changed; a fresh NumPy-only environment
   has not been dynamically tested here.
8. The actual `nvdev` installation has `direct_url.json` with `editable: true`
   and the repository URL, plus `__editable__.curegot-0.1.0.pth` and its finder
   in site-packages. That installed finder explains fresh-cwd imports.
9. Production imports are package-relative (`._internal_torch`, `.backward`,
   `._backward_core`) and require no repository-root shims or cwd paths. The
   validator runs its full checks in `/tmp` using `python -I -c`, removes
   `PYTHONPATH`, and rejects root-shim imports via a meta-path finder.
10. Historical Stage 5 documents contain stale “Workload C/example pending”
    statements, references to a sandbox-only `curegot.curegot` placeholder and
    a withdrawn setup file that are not descriptions of this checkout.
    `STAGE5_PRODUCTION_API.md` changes cwd to `/tmp` then gives relative test
    commands without returning to the repo. “No reinstall needed” applies to
    the existing editable install, not an old regular wheel. Historical
    documents are preserved; use the commands below for release acceptance.

No confirmed Python-file inclusion or extension-naming packaging bug was found.
Review concerns remain: `setup.py` can silently skip torch extension setup via
its broad exception handler, runtime imports can hide failures, and native ABI /
shared-library portability needs real wheel acceptance. A future, separately
reviewed minimal improvement could report the caught torch build/load error;
no such production/build change is necessary for this documentation stage.
Public docstrings already document the main contract and are left unchanged.
The autograd module's blanket “raises on second differentiation” wording is
stronger than the historical Stage 5 note about linear losses: the supported
contract is simply **no double backward**, without a guaranteed exception form.

## Current environment regression (no installation or native rebuild)

The shell's default Python is base Conda, not `nvdev`. On this machine the
compiled extensions need Conda's newer `libstdc++`; without its library path,
importing torch first can produce `GLIBCXX_3.4.31 not found` from the system
library. Use this process-local shell configuration:

```bash
cd /home/benben_cd/regot-cuda
(
  export PATH=/home/benben_cd/miniconda3/envs/nvdev/bin:$PATH
  export LD_LIBRARY_PATH=/home/benben_cd/miniconda3/envs/nvdev/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  python validate_stage6a_release.py
  python validate_stage5_api.py
  python validate_stage5_workload.py
  python examples/differentiable_splr_ot.py
)
```

The validator fails if installation metadata, API, CUDA, convergence or numerical
checks fail; unavailable GPU is not a successful skip. Run on the host if the
sandbox cannot access the GPU. Stage 4 full regression is unnecessary for this
docs/validation-only change.

## Build a wheel using existing dependencies in temporary environments

These commands are a **manual plan, not already executed**. They use unique
`/tmp/curegot-stage6a-*` directories and do not install into, delete, or alter
`nvdev`. They reuse its Python/PyTorch/runtime dependencies through
`--system-site-packages`, so this is an isolated package-install acceptance
with shared dependencies, **not a completely independent dependency environment**.
A second machine/container is still needed for portability acceptance.

The build runs in a temporary source copy because the custom NumPy builder
writes `.o` files alongside `src/*.cu`. The commands copy tracked sources at
HEAD plus the Stage 6A deliverables; review/commit other intended source edits
before using this procedure. Do not use the historical `nvdev` environment
exports as portable requirements: they contain a machine-specific prefix/local
curegot URL, and the installed requests version differs from the build constraint.

```bash
set -e
release_repo=/home/benben_cd/regot-cuda
release_base=/home/benben_cd/miniconda3/envs/nvdev
release_work=$(mktemp -d /tmp/curegot-stage6a-XXXXXX)
mkdir "$release_work/source"
git -C "$release_repo" archive HEAD | tar -x -C "$release_work/source"
cp "$release_repo/README.md" "$release_repo/validate_stage6a_release.py" \
   "$release_repo/STAGE6A_CLEAN_INSTALL.md" "$release_work/source/"

"$release_base/bin/python" -m venv --system-site-packages "$release_work/build-env"
"$release_base/bin/python" -m venv --system-site-packages "$release_work/install-env"
export CUDA_HOME="$release_base"
export PATH="$release_base/bin:$PATH"
export LD_LIBRARY_PATH="$release_base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$release_base/bin/x86_64-conda-linux-gnu-cc"
export CXX="$release_base/bin/x86_64-conda-linux-gnu-c++"
export CCCL_INCLUDE_DIR="$release_repo/cccl-v3.3.3/include"
test -f "$CCCL_INCLUDE_DIR/cuda/std/version"
test -f "$CUDA_HOME/include/cudss.h"
test -f "$CUDA_HOME/lib/libcudss.so"
"$CUDA_HOME/bin/nvcc" --version
"$release_work/build-env/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'

# Only small build dependencies are installed locally; reuse existing torch.
"$release_work/build-env/bin/python" -m pip install build \
  'setuptools>=80.0' 'pybind11~=3.0.0' 'requests~=2.32.0'
cd "$release_work/source"
# Default build creates an sdist and builds the wheel from that sdist.
# --no-isolation reuses the prepared environment; this DOES compile native code.
"$release_work/build-env/bin/python" -m build --no-isolation

"$release_work/build-env/bin/python" - <<'PY'
from pathlib import Path
from zipfile import ZipFile
wheels = list(Path('dist').glob('*.whl'))
assert len(wheels) == 1, wheels
with ZipFile(wheels[0]) as z:
    names = z.namelist()
    for name in ('__init__.py', '_cuda_loader.py', '_backward_core.py',
                 'backward.py', 'autograd.py'):
        assert 'curegot/' + name in names, name
    for stem in ('_internal_numpy', '_internal_torch'):
        assert any(n.startswith('curegot/' + stem) and n.endswith('.so')
                   for n in names), stem
    assert not any(n.startswith('curegot/curegot/') for n in names)
print('PASS wheel Python/native contents:', wheels[0])
PY

# Force a LOCAL install even though shared site-packages exposes editable curegot.
# --no-deps prevents re-downloading PyTorch/CUDA and reuses existing runtimes.
"$release_work/install-env/bin/python" -m pip install --no-deps --ignore-installed dist/*.whl
cd /tmp
"$release_work/install-env/bin/python" -I - <<'PY'
import sys
from pathlib import Path
import curegot
p = Path(curegot.__file__).resolve()
assert p.is_relative_to(Path(sys.prefix).resolve()), p
print('Wheel-local import:', p)
PY
"$release_work/install-env/bin/python" "$release_work/source/validate_stage6a_release.py"
"$release_work/install-env/bin/python" "$release_work/source/examples/differentiable_splr_ot.py"
printf 'Artifacts retained for review: %s\n' "$release_work"
```

For editable source acceptance in a separately prepared environment, replace the
wheel-install step with `python -m pip install --no-build-isolation -e .` from
the intended source directory. This can also compile native extensions; it is
not equivalent to installing a prebuilt wheel. The isolated Stage 6A validator
accepts editable installs but the wheel-local path assertion deliberately does not.

For a fully independent installation, first provision Python >=3.11, compatible
CUDA Toolkit >=13, cuDSS development files, a compatible C++ toolchain and
CUDA-enabled PyTorch >=2.10. Install the project's declared runtime dependencies
(`numpy>=1.23`, `cuda-pathfinder>=1.5`, `nvidia-cuda-runtime>=13`,
`nvidia-cudss-cu13>=0.7`) and build dependencies, then use the same source build,
wheel inspection and isolated validation steps. An ordinary wheel installation
without `--no-deps` installs the declared runtime dependencies; PyTorch must be
provisioned separately. Driver support and torch/CUDA/C++ ABI compatibility are
external prerequisites, not validated by package metadata alone.

## Validation recorded in this change (2026-09-13)

Host GPU: RTX 5060 Ti; Python 3.12 in `nvdev`; PyTorch 2.13.0+cu130.

| Executed check | Result |
|---|---|
| `validate_stage6a_release.py` | PASS; public/manual gradient relative error 5.013e-15 |
| `validate_stage5_api.py` | PASS, 13 groups; gradient relative error 9.372e-16 |
| `validate_stage5_workload.py` | PASS, 5 groups; CUDA memory growth 0 bytes |
| `examples/differentiable_splr_ot.py` | PASS; loss -0.02374479 to -0.9294546 |
| Python syntax, documentation bash syntax, `git diff --check` | PASS |
| Stage 4 full regression | NOT RUN: production implementation unchanged |
| New environment installation, sdist/wheel build and wheel installation | NOT RUN: manual procedure above |

Initial attempts exposed two setup/harness issues: the default interpreter/library
search did not provide the required Conda C++ runtime for torch-first imports;
setting the process library path fixed the existing regression. The new smoke
check initially requested marginal accuracy tighter than obtained with default
forward tolerance; README and validator now explicitly request `tol=1e-8`, as
in the existing example. Production code and existing tests were not changed.
`torch.compile` ran with the expected tracing warning/graph breaks; TorchScript
reported `NotSupportedError`. Neither is promoted to a supported interface.
