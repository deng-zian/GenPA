# Minimal policy inputs

These small inputs are drawn from the simulation training data. They contain an RGB canvas, proprioception, an exact-prompt Qwen context, and its mask. They allow a checkpoint-loading and action-inference check without downloading the full training dataset or building a new text cache. Context values retain their original BF16 bit pattern.

The examples are installation checks, not evaluation episodes or success-rate measurements.
