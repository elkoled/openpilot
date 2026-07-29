## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

### Precompiled USB-GPU big model

The big driving model is never compiled by SCons or during device boot. Its
compiled tinygrad pickle is built by the `big model artifact` GitHub Actions
workflow on a self-hosted ARM64 runner labeled `usbgpu`. The workflow:

1. compiles and numerically round-trip checks the model on the production USB
   GPU;
2. splits the pickle into 45 MiB GitHub Release assets;
3. verifies every chunk, the reassembled SHA-256, and a real target-device load;
4. publishes the assets under the content-derived
   `big-model-<artifact_id>` release tag.

`big_model_artifact.json` binds the source tree to one artifact. Any model,
tinygrad, compiler, serialization, or preprocessing change alters the
fingerprint and requires a new artifact:

```sh
python3 openpilot/selfdrive/modeld/big_model_artifact.py write-descriptor
```

After that descriptor is reviewed and committed to `gpu-nightly`, dispatch the
workflow from that branch with `publish=true`. The hardware build/publish job is
gated to `gpu-nightly`, and existing release tags are never replaced.

On USB-GPU devices, `updated` downloads and fully verifies the versioned
artifact into `/data/model_cache/openpilot` before finalizing the software
update. Interrupted downloads resume at the next missing chunk. At boot,
`modeld` only opens the already-local artifact; missing or invalid artifacts
fall back to the small model.
