"""LoRA adapter injection and management utilities.

This module provides utilities to inject, manage, and remove LoRA adapters
from BERT models, enabling flexible fine-tuning strategies for Kaggle competitions.
"""

import mlx.core as mx
import mlx.nn as nn

from ..quantization_utils import QuantizedLinear
from .config import LoRAConfig, QLoRAConfig
from .layers import LoRALinear, MultiLoRALinear, QLoRALinear


class LoRAAdapter:
    """Manages LoRA adapter injection and removal for BERT models.

    This class provides methods to:
    - Inject LoRA adapters into specific layers
    - Remove adapters and restore original layers
    - Merge adapters for inference
    - Save/load adapter weights separately
    """

    def __init__(self, model: nn.Module, config: LoRAConfig | QLoRAConfig):
        """Initialize LoRA adapter manager.

        Args:
            model: The BERT model to adapt
            config: LoRA configuration
        """
        self.model = model
        self.config = config
        self.original_modules = {}
        self.lora_modules = {}
        self.injected_modules = set()

    def inject_adapters(self, verbose: bool = False) -> dict[str, int]:
        """Inject LoRA adapters into the model.

        Args:
            verbose: Print injection details

        Returns:
            Dictionary mapping module names to number of parameters added
        """
        stats = {}

        for name, module in self.model.named_modules():
            if self._should_inject_lora(name, module):
                new_module = self._create_lora_module(name, module)
                if new_module is not None:
                    # Store original module
                    self.original_modules[name] = module

                    # Replace with LoRA module
                    self._replace_module(name, new_module)
                    self.lora_modules[name] = new_module
                    self.injected_modules.add(name)

                    # Track parameters added
                    if hasattr(new_module, "trainable_parameters"):
                        params_added = new_module.trainable_parameters
                    else:
                        params_added = self._count_lora_parameters(new_module)
                    stats[name] = params_added

                    if verbose:
                        print(
                            f"Injected LoRA into {name}: +{params_added:,} parameters"
                        )

        # Freeze non-LoRA parameters
        self._freeze_non_lora_parameters()

        if verbose:
            total_params = sum(stats.values())
            print(f"\nTotal LoRA parameters: {total_params:,}")
            print(f"Injected into {len(stats)} modules")

        return stats

    def remove_adapters(self, restore_original: bool = True) -> None:
        """Remove all LoRA adapters from the model.

        Args:
            restore_original: Whether to restore original modules
        """
        for name in list(self.injected_modules):
            if restore_original and name in self.original_modules:
                self._replace_module(name, self.original_modules[name])

        self.injected_modules.clear()
        self.lora_modules.clear()
        if restore_original:
            self.original_modules.clear()

    def merge_adapters(self) -> dict[str, bool]:
        """Merge LoRA weights into base model for inference.

        Returns:
            Dictionary mapping module names to merge success status
        """
        merge_status = {}

        for name, lora_module in self.lora_modules.items():
            if hasattr(lora_module, "merge_weights"):
                try:
                    merged = lora_module.merge_weights()
                    self._replace_module(name, merged)
                    merge_status[name] = True
                except Exception as e:
                    print(f"Failed to merge {name}: {e}")
                    merge_status[name] = False
            else:
                merge_status[name] = False

        # Clear LoRA tracking after merge
        self.injected_modules.clear()
        self.lora_modules.clear()

        return merge_status

    def get_lora_state_dict(self) -> dict[str, mx.array]:
        """Get state dict containing only LoRA parameters.

        Returns:
            Dictionary of LoRA parameters
        """
        lora_state = {}

        for name, module in self.model.named_modules():
            if name in self.lora_modules:
                # Get LoRA-specific parameters
                lora_params = self._extract_lora_params(name, module)
                lora_state.update(lora_params)

        return lora_state

    def load_lora_state_dict(self, state_dict: dict[str, mx.array]) -> None:
        """Load LoRA parameters from state dict.

        Args:
            state_dict: Dictionary of LoRA parameters
        """
        for name, param in state_dict.items():
            # Extract module name and parameter name
            module_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]

            if module_name in self.lora_modules:
                module = self.lora_modules[module_name]
                if hasattr(module, param_name):
                    setattr(module, param_name, param)

    def _should_inject_lora(self, name: str, module: nn.Module) -> bool:
        """Check if LoRA should be injected into a module.

        Args:
            name: Module name
            module: Module instance

        Returns:
            Whether to inject LoRA
        """
        # Only inject into Linear layers
        if not isinstance(module, nn.Linear | QuantizedLinear):
            return False

        # Check against target modules
        if not self.config.should_apply_lora(name):
            return False

        # Don't inject if already has LoRA
        return not isinstance(module, LoRALinear | QLoRALinear | MultiLoRALinear)

    def _create_lora_module(
        self, name: str, module: nn.Module
    ) -> LoRALinear | QLoRALinear | None:
        """Create a LoRA module to replace the original.

        Args:
            name: Module name
            module: Original module

        Returns:
            LoRA module or None if creation fails
        """
        if isinstance(module, nn.Linear):
            in_features = module.weight.shape[1]
            out_features = module.weight.shape[0]

            # Get layer-specific config if available
            layer_config = self.config.get_layer_config(name)

            # Create config for this specific layer
            if isinstance(self.config, QLoRAConfig):
                layer_lora_config = QLoRAConfig(
                    **{**self.config.__dict__, **layer_config}
                )
                return QLoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    config=layer_lora_config,
                    base_layer=module,
                )
            else:
                layer_lora_config = LoRAConfig(
                    **{**self.config.__dict__, **layer_config}
                )
                return LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    config=layer_lora_config,
                    base_layer=module,
                )

        elif isinstance(module, QuantizedLinear):
            # For already quantized layers, use QLoRA
            in_features = module.in_features
            out_features = module.out_features

            if not isinstance(self.config, QLoRAConfig):
                # Convert LoRA config to QLoRA config
                qlora_config = QLoRAConfig(**self.config.__dict__)
            else:
                qlora_config = self.config

            return QLoRALinear(
                in_features=in_features,
                out_features=out_features,
                config=qlora_config,
                base_layer=module,
            )

        return None

    def _replace_module(self, name: str, new_module: nn.Module) -> None:
        """Replace a module in the model.

        Args:
            name: Full module name (dot-separated)
            new_module: New module to insert
        """
        # MLX's named_modules correctly generates names like "bert.encoder_layers.0.dense"
        # We can use MLX's built-in module update functionality
        modules_dict = dict(self.model.named_modules())

        # Verify the module exists
        if name not in modules_dict:
            raise ValueError(f"Module {name} not found in model")

        # Update the module using MLX's module update
        self.model.update({name: new_module})

    def _freeze_non_lora_parameters(self) -> None:
        """Freeze all non-LoRA parameters in the model."""
        # In MLX, we freeze modules, not individual parameters
        for name, module in self.model.named_modules():
            # Skip the model itself
            if name == "":
                continue

            # Check if module contains LoRA parameters
            is_lora_module = any(
                hasattr(module, lora_attr)
                for lora_attr in ["lora_A", "lora_B", "lora_bias", "magnitude"]
            )

            # Check if module is in modules to save
            in_modules_to_save = any(
                module_name in name for module_name in self.config.modules_to_save
            )

            # Freeze non-LoRA modules
            if not is_lora_module and not in_modules_to_save:
                # Only freeze if it's a leaf module (no children)
                if not list(module.children()):
                    module.freeze()

    def _count_lora_parameters(self, module: nn.Module) -> int:
        """Count LoRA parameters in a module.

        Args:
            module: Module to count parameters in

        Returns:
            Number of LoRA parameters
        """
        count = 0
        for name, param in module.named_parameters():
            if any(n in name for n in ["lora_A", "lora_B", "lora_bias", "magnitude"]):
                count += param.size
        return count

    def _extract_lora_params(
        self, module_name: str, module: nn.Module
    ) -> dict[str, mx.array]:
        """Extract LoRA parameters from a module.

        Args:
            module_name: Name of the module
            module: Module instance

        Returns:
            Dictionary of LoRA parameters
        """
        params = {}

        if hasattr(module, "lora_A"):
            params[f"{module_name}.lora_A"] = module.lora_A
        if hasattr(module, "lora_B"):
            params[f"{module_name}.lora_B"] = module.lora_B
        if hasattr(module, "lora_bias") and module.lora_bias is not None:
            params[f"{module_name}.lora_bias"] = module.lora_bias
        if hasattr(module, "magnitude"):
            params[f"{module_name}.magnitude"] = module.magnitude

        return params


class MultiAdapterManager:
    """Manages multiple LoRA adapters for a single model.

    Useful for:
    - Multi-task learning with different adapters per task
    - Ensemble methods using multiple adapters
    - A/B testing different adapter configurations
    """

    def __init__(self, model: nn.Module):
        """Initialize multi-adapter manager.

        Args:
            model: Base BERT model
        """
        self.model = model
        self.adapters = {}
        self.active_adapter = None

    def add_adapter(
        self, name: str, config: LoRAConfig | QLoRAConfig, activate: bool = False
    ) -> LoRAAdapter:
        """Add a new adapter configuration.

        Args:
            name: Name for the adapter
            config: LoRA configuration
            activate: Whether to activate immediately

        Returns:
            The created LoRA adapter
        """
        adapter = LoRAAdapter(self.model, config)
        self.adapters[name] = adapter

        if activate:
            self.activate_adapter(name)

        return adapter

    def activate_adapter(self, name: str) -> dict[str, int]:
        """Activate a specific adapter.

        Args:
            name: Name of adapter to activate

        Returns:
            Injection statistics
        """
        if self.active_adapter is not None:
            # Remove current adapter
            self.adapters[self.active_adapter].remove_adapters()

        if name in self.adapters:
            stats = self.adapters[name].inject_adapters()
            self.active_adapter = name
            return stats
        else:
            raise ValueError(f"Adapter '{name}' not found")

    def deactivate_adapter(self) -> None:
        """Deactivate the current adapter."""
        if self.active_adapter is not None:
            self.adapters[self.active_adapter].remove_adapters()
            self.active_adapter = None

    def get_adapter(self, name: str) -> LoRAAdapter | None:
        """Get a specific adapter.

        Args:
            name: Adapter name

        Returns:
            LoRA adapter or None
        """
        return self.adapters.get(name)

    def save_all_adapters(self, directory: str) -> None:
        """Save all adapter weights to directory.

        Args:
            directory: Directory to save adapters
        """
        import os

        os.makedirs(directory, exist_ok=True)

        for name, adapter in self.adapters.items():
            state_dict = adapter.get_lora_state_dict()
            # Save using safetensors format
            import safetensors

            safetensors.mlx.save_file(
                state_dict, os.path.join(directory, f"{name}_adapter.safetensors")
            )

    def load_adapter_weights(self, name: str, path: str) -> None:
        """Load adapter weights from file.

        Args:
            name: Adapter name
            path: Path to weights file
        """
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found")

        import safetensors

        state_dict = safetensors.mlx.load_file(path)
        self.adapters[name].load_lora_state_dict(state_dict)
