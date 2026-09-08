# Chestnut compile CI

The first runner compiles the big driving model on `comma-de2e7866`, using the
`mici-chestnut-ci` lock, existing Jenkins SSH image and `id_rsa` credential.
It calls openpilot's `compile_modeld.py` with the Chestnut flags from SConscript.
The compiler includes capture, replay and serialization correctness checks.
There is no separate inference loop, benchmark job or report framework.
Compiler output appears directly in the Jenkins console. A compiler failure,
timeout or missing or empty output fails the build.

The bench must be an offroad MICI with Chestnut connected, autostart disabled
and no manager or modeld running. The runner fetches the exact commit into
`/data/chestnut-ci-workspace/source` and downloads only the big model, with one
LFS transfer and one retry. Compilation has a 40 minute timeout inside a one
hour remote session. Cleanup stops the remote process group before releasing
the device lock and removes the compiled artifact.

`openpilot-chestnut-dev` runs only this stage from `elkoled/openpilot`, branch
`chestnut-jenkins-ci`. The same stage is included in the regular pipeline for
future upstream integration. Disable the development job after migration.
The dedicated checkout checks its origin URL, so rename it before changing
from the fork to commaai. Device `comma-95940f7f` is not used by this runner.
Registered device resources are eligible for the existing hourly reboot job.

Local checks:

```sh
python3 -m unittest discover -s tools/ci/chestnut -p 'test_*.py'
bash -n tools/ci/chestnut/remote.sh
bash -n tools/ci/chestnut/cancel.sh
```
