"""
MLX Compatibility Layer

This module provides compatibility fixes for different MLX versions to ensure
mlx-embeddings works correctly across various MLX releases.
"""

import mlx.core as mx
from loguru import logger


def patch_mlx_embeddings_compatibility():
    """
    Patch mlx-embeddings to work with different MLX versions.

    The main issue is that the astype() method signature changed between
    MLX versions, and mlx-embeddings was written for an older API.
    """
    try:
        # Import mlx-embeddings modernbert module
        import mlx_embeddings.models.modernbert as modernbert_module

        # Get the original ModernBertModel class
        original_model = modernbert_module.ModernBertModel

        # Patch the _update_attention_mask method
        def patched_update_attention_mask(self, attention_mask):
            """
            Patched version of _update_attention_mask to use correct astype() signature.
            """
            # Get the dtype from the attention mask
            dtype = attention_mask.dtype

            # Create sliding window mask
            sliding_window_mask = mx.full(
                attention_mask.shape + (attention_mask.shape[-1],),
                0.0,
                dtype=mx.float32,
            )

            # Apply sliding window if configured
            if hasattr(self, "sliding_window") and self.sliding_window is not None:
                seq_len = attention_mask.shape[-1]
                sliding_window = min(self.sliding_window, seq_len)

                # Create sliding window mask
                mask = mx.ones((seq_len, seq_len), dtype=mx.float32)
                for i in range(seq_len):
                    start = max(0, i - sliding_window // 2)
                    end = min(seq_len, i + sliding_window // 2 + 1)
                    mask = mx.where(
                        (mx.arange(seq_len) >= start) & (mx.arange(seq_len) < end),
                        mask,
                        -1e9,
                    )
                sliding_window_mask = mask

            # Convert attention mask to proper format
            global_attention_mask = mx.where(attention_mask == 0, -1e9, 0.0)

            # Use correct astype signature for MLX 0.24.2
            # The new signature expects mlx.core.Dtype objects
            if hasattr(mx, "int32"):
                target_dtype = mx.int32
            else:
                target_dtype = mx.Dtype.int32

            try:
                # Try the new astype signature
                return (
                    global_attention_mask.astype(target_dtype),
                    sliding_window_mask.astype(target_dtype),
                )
            except Exception as e:
                logger.warning(f"Failed to use new astype signature: {e}")
                # Fall back to original method if possible
                return (
                    global_attention_mask.astype(dtype),
                    sliding_window_mask.astype(dtype),
                )

        # Replace the method
        original_model._update_attention_mask = patched_update_attention_mask

        logger.info("Successfully patched mlx-embeddings for MLX compatibility")
        return True

    except Exception as e:
        logger.warning(f"Failed to patch mlx-embeddings compatibility: {e}")
        return False


def apply_mlx_patches():
    """
    Apply all MLX compatibility patches.

    This function should be called before using mlx-embeddings to ensure
    compatibility with the current MLX version.
    """
    logger.info("Applying MLX compatibility patches...")

    success = patch_mlx_embeddings_compatibility()

    if success:
        logger.info("MLX compatibility patches applied successfully")
    else:
        logger.warning("Some MLX compatibility patches failed")

    return success
