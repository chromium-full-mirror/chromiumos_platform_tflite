# ChromeOS TFLite

This repository hosts the core ChromeOS TFLite components, enabling on-device
machine learning (ODML) workloads accelerated by NPU.

The corresponding ebuild can be found at:
[tensorflow-9999.ebuild](https://chromium.googlesource.com/chromiumos/overlays/chromiumos-overlay/+/main/sci-libs/tensorflow/tensorflow-9999.ebuild)

## TensorFlow Patch Management

Patches are stored in the `patch/` directory and explicitly listed in
`WORKSPACE.bazel`. A helper script, `./script/patcher.py`, is included to
facilitate patch management within a TFLite workspace.

The typical workflow:

1.  **Eject (Download) TensorFlow Source Code**

    Download the TensorFlow source code into a local git repository with patches
    applied as individual commits:

    ```bash
    ./script/patcher.py eject
    ```

    This creates a new local git repository at `tensorflow/`.

2.  **Modify the TensorFlow Repository**

    Make changes to the `tensorflow/` repository as needed, following standard
    git workflows. Optionally, include a `PATCH_NAME=` tag in commit messages to
    specify the filename of the corresponding patch.

3.  **Seal the Repository**

    Regenerate the patch files and update the `WORKSPACE.bazel` file:

    ```bash
    ./script/patcher.py seal
    ```

    This updates the patches in the `patch/` directory and reflects the changes
    in `WORKSPACE.bazel`.

It's preferred to submit changes to upstream TensorFlow first and cherry-pick
them as patches. This helps minimize divergence and makes TensorFlow updates
easier.
