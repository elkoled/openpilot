# Chestnut Jenkins example

This pipeline targets `elkoled/openpilot`'s sunnypilot-based master at
`de197ba6fa`. It compiles the upstream big-model ONNX on a MICI + powered
Chestnut, runs the compiler's JIT serialization/correctness checks, then loads
the resulting artifact and runs 20 synthetic policy inferences on USB+AMD.
Select 100 or 1000 iterations for a longer smoke test.

This is model CI, not the full sunnypilot model manager or a drive/power-cycle
stress test. The compiler tests the MICI image warp; the inference loop feeds
synthetic warped images to the policy. Timing is reported, not gated yet.
No fallback to the small model is allowed. No fleet logs are downloaded.

## Agent setup

Use a dedicated offroad MICI/Chestnut pair with stable 12 V GPU power and the
fork's compatible AGNOS/Chestnut firmware already installed. Provision a Jenkins
agent as the `comma` user with label **mici-chestnut-ci**, **one executor**, and
remote workspace under `/data/jenkins`. Reserve that label for this job. Install
the Java runtime required by your Jenkins controller and expose the existing
AGNOS Python 3.12 environment to the agent's PATH. It needs the fork's Python
dependencies, including numpy, zstandard and tinygrad's compiler dependencies,
plus git, git-lfs, taskset, and at least 12 GiB free workspace storage.

Disable the normal openpilot launcher using your existing bench provisioning
procedure. Verify `IsOnroad` is `0` and create `/data/chestnut-ci-bench` to mark
this as a dedicated test device. The job fails if manager/modeld is still
running; it does not stop services or alter the boot configuration for you.

The agent needs network access to GitHub, the fork's configured GitLab LFS
endpoint and any tinygrad firmware dependencies absent from `/lib/firmware`.
Only tinygrad's pinned submodule and the big ONNX model are fetched. The model
is approximately 1.8 GB; later builds reuse Git LFS's local object cache.

## Create the job

1. Install/enable Pipeline (including Declarative), Git and JUnit
   plugins on Jenkins.
2. Create a **Pipeline** job named `elkoled-chestnut-ci`.
3. Choose **Pipeline script from SCM → Git**.
4. Repository: `https://github.com/elkoled/openpilot.git`.
5. Branch: `*/chestnut-jenkins-example`.
6. Script path: `tools/ci/chestnut/Jenkinsfile`.
7. Save and select **Build Now**. After the first run, use **Build with Parameters**
   to change `SMOKE_RUNS`.

The build records the checked-out commit, compile log, JUnit results and
inference timing JSON. Large compiled weights are not archived and are removed
after each build. The compile is always fresh, takes at most 40 minutes, and
the entire job has a 60-minute timeout. Builds are serialized with no automatic
test retries. Jenkins abort handling terminates the agent's build processes;
verify the executor and compiler processes are idle before reusing a cancelled
bench. This example does not automatically trigger on untrusted PRs.

## Local validation (no hardware)

```sh
python3 -m unittest discover -s tools/ci/chestnut -p 'test_*.py'
python3 tools/ci/chestnut/run.py --help
```

These checks validate orchestration and failure reporting only. A green
hardware run is still required before treating this pipeline as operational.
