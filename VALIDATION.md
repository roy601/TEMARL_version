# Local validation

Validated on Windows with Python 3.13.3, PyTorch 2.10.0+cpu,
NumPy 2.4.2 and SciPy 1.18.1.

| Check | Outcome |
|---|---|
| Environment mechanics, determinism and observation locality | 27/27 passed |
| Paper equations, encoder masks/gradients/checkpoints, PPO identity, IPPO critic locality, data exclusion and reporting | 13/13 passed |
| Standalone relocation | Both suites passed from a temporary copy outside the repository, with an unrelated working directory |
| CPU end-to-end execution | Six requested combinations and two no-history controls completed |
| Report generation | Markdown, CSV and JSON generated from smoke outputs |
| Completed-cell resume | All eight completed cells skipped; report regenerated |
| Python compilation | Passed |
| Frozen source/data integrity | Passed mandatory hash checks |

The final CPU execution check used `--smoke --device cpu --output
results/smoke_final`. Each cell used only 16 RL environment steps and eight
test episodes. These numbers test execution, NOT learning or model quality.
Earlier smoke directories are development checks, not research results.

No full experiment, convergence study, new scientific performance gate or
GPU execution has been completed for this package. This machine has CPU-only
PyTorch. CUDA support is implemented but requires validation on lab hardware
with a compatible CUDA-enabled PyTorch installation. Package ranges are not
a guarantee that every allowed dependency version was tested.

The distributable ZIP excludes generated results, caches and virtual
environments. It includes the source, grounding data, reference papers,
requirements, tests and documentation. Install dependencies on the lab PC.

Original repository code and historical experiment outputs were not changed.
