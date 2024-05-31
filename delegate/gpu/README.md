# Stable GPU Delegate

This encapsulates the existing GPU delegate using the stable delegate
interface, allowing us to use it as a reference for our stable delegate-related
tooling.

Please note that the implementation of the GPU delegate is not ABI stable, as
it relies on non-opaque APIs internally.
