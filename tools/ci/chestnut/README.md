# Chestnut CI on comma's Jenkins

This branch is based on commaai/openpilot at `25d9d41c90`. It uses the existing
Jenkins server, `ghcr.io/commaai/alpine-ssh`, the `id_rsa` credential and Lockable
Resources. No Jenkins agent or Java installation is needed on either device.

| Device | Jenkins resource label | Purpose |
| --- | --- | --- |
| comma-de2e7866 | mici-chestnut-ci | Compile and short model CI |
| comma-95940f7f | mici-chestnut-stress | Reserved for the later drive/power-cycle job |

## First job

1. Verify both benches have stable Chestnut power. The first bench must have
   `/data/disable_openpilot_autostart` present and no manager/modeld running.
   This is the existing bench provisioning; this pipeline does not change it.
2. In **Manage Jenkins > Lockable Resources**, add `comma-de2e7866`, label
   `mici-chestnut-ci`, and leave **Reserved by** empty. Registering a `comma-*`
   resource also includes it in the existing hourly maintenance/reboot job.
3. Create **Multibranch Pipeline** `openpilot-chestnut-dev`. Add GitHub source
   owner `elkoled`, repository `openpilot`, using the existing GitHub connection
   if required. Set the branch-name regex to exactly `^chestnut-jenkins-ci$`.
   Keep script path `Jenkinsfile`. Do not copy the one-day aged-branch filter.
4. Save and scan. The branch's root Jenkinsfile runs only the Chestnut stage.
   Saving/scanning may start the build automatically.
5. Open the build's Console Output and Test Results. A passing build must compile
   the big ONNX, check JIT capture/serialization correctness, then execute 20
   synthetic full-model inferences (including image warp) on USB+AMD. Non-finite
   outputs or a non-AMD artifact fail the build. Timing is reported, not gated.

The pipeline locks device 1 and checks its hostname. Its checkout is separate
from the installed openpilot: `/data/chestnut-ci-workspace/source`. It fetches the
exact fork SHA and pinned tinygrad submodule. Only big-model LFS objects are
fetched (about 1.8 GB initially), with one transfer and bounded retries. There
are no fleet-data downloads, firmware flashes, or changes to boot configuration.
Reports include the commit, hostname, compile log, JUnit XML and timing JSON.
The hardware session has a one-hour limit, with a 40-minute compiler timeout.

To test upstream after merging, the same stage is already included alongside
existing tests in the root Jenkinsfile. It accepts either commaai's or elkoled's
HTTPS repository URL. A workspace initialized for one URL deliberately refuses
the other: archive/rename that dedicated CI checkout once during migration so
it can initialize from commaai. Disable the development job after migration.

Device 2 is not exercised by this initial pipeline. Continuous drives,
modeld restart tests and physical power cycling require a separate bounded job
and an identified power-control mechanism. Keep it on a separate resource lock.

## Local checks

```sh
python3 -m unittest discover -s tools/ci/chestnut -p 'test_*.py'
bash -n tools/ci/chestnut/remote.sh
```

Local tests check orchestration, limits and failure reporting. They do not
substitute for a successful hardware build.
