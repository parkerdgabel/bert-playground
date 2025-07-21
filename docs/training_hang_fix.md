# MLX Training Hang Fix Documentation

## Problem Description

The training was hanging at batch 0 when using ModernBERT with the MLX framework. The hang occurred during the first gradient computation step, making training impossible.

## Root Cause

The issue was caused by **lazy evaluation buildup** in the MLX dataloader. MLX uses lazy evaluation, meaning operations are not computed immediately but are recorded in a computation graph. When the dataloader created MLX arrays in a separate prefetch thread without evaluating them, these unevaluated arrays were passed to the training loop. When gradient computation was attempted on these unevaluated arrays, it created an exponentially large computation graph that caused the hang.

This is a known issue in MLX, documented in [issue #451](https://github.com/ml-explore/mlx/issues/451).

## Solution

The fix was to force evaluation of batch arrays in the MLXDataLoader before they are yielded or put in the prefetch queue. This prevents the lazy evaluation buildup.

### Code Changes

In `data/loaders/mlx_loader.py`, we added `mx.eval()` calls in two places:

1. **In the prefetch worker thread** (lines 195-202):
```python
# Collate into batch
batch = self._collate_samples(samples)

# CRITICAL: Force evaluation of batch arrays to prevent lazy evaluation buildup
# This prevents hanging when computing gradients on unevaluated arrays
# See: https://github.com/ml-explore/mlx/issues/451
if batch:
    # Evaluate all arrays in the batch
    arrays_to_eval = [v for v in batch.values() if isinstance(v, mx.array)]
    if arrays_to_eval:
        mx.eval(*arrays_to_eval)

# Put in queue (blocks if full)
self.prefetch_queue.put(batch)
```

2. **In the non-prefetch iteration** (lines 145-152):
```python
# Collate into batch
batch = self._collate_samples(samples)

# CRITICAL: Force evaluation of batch arrays to prevent lazy evaluation buildup
# This prevents hanging when computing gradients on unevaluated arrays
# See: https://github.com/ml-explore/mlx/issues/451
if batch:
    # Evaluate all arrays in the batch
    arrays_to_eval = [v for v in batch.values() if isinstance(v, mx.array)]
    if arrays_to_eval:
        mx.eval(*arrays_to_eval)

logger.debug(f"Yielding batch {batch_idx} with keys: {list(batch.keys())}")
yield batch
```

## Impact

This fix ensures that:
1. Training no longer hangs at batch 0
2. Gradient computation works correctly with dataloader batches
3. Both prefetch and non-prefetch modes work properly
4. Performance is maintained while preventing lazy evaluation issues

## Testing

The fix was verified through:
1. Diagnostic scripts that isolated the issue to dataloader batch creation
2. Successful training runs with the fixed dataloader
3. Both small and large batch sizes work correctly

## Additional Notes

- The mx.eval() in base.py after gradient computation is still important for preventing buildup during training
- Compilation issues with int64 types are a separate issue that can be addressed independently
- This fix is critical for any MLX dataloader implementation that uses threading or async operations