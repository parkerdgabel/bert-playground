"""Builder for creating models with LoRA and QLoRA adapters."""

from typing import Any

from core.bootstrap import get_service
from core.ports.compute import ComputeBackend, Module
from loguru import logger

from ..lora import LoRAAdapter, LoRAConfig, MultiAdapterManager, QLoRAConfig, get_lora_preset


class LoRABuilder:
    """Builder for models with LoRA and QLoRA adapters."""

    def __init__(self):
        self.compute_backend = get_service(ComputeBackend)

    def build_lora_model(
        self,
        base_model: Module,
        lora_config: LoRAConfig | QLoRAConfig | str | dict | None = None,
        inject_adapters: bool = True,
        verbose: bool = False,
    ) -> tuple[Module, LoRAAdapter]:
        """Build a model with LoRA adapters.
        
        Args:
            base_model: Base model to add LoRA to
            lora_config: LoRA configuration (config object, preset name, or dict)
            inject_adapters: Whether to inject LoRA adapters immediately
            verbose: Print injection details
            
        Returns:
            Tuple of (model, lora_adapter)
        """
        logger.info("Building model with LoRA adapters")
        
        # Process LoRA config
        if lora_config is None:
            lora_config = LoRAConfig()  # Default config
        elif isinstance(lora_config, str):
            # Load preset
            lora_config = get_lora_preset(lora_config)
            logger.info(f"Using LoRA preset: {lora_config}")
        elif isinstance(lora_config, dict):
            # Determine if it's QLoRA based on config
            if any(k.startswith("bnb_") for k in lora_config):
                lora_config = QLoRAConfig(**lora_config)
            else:
                lora_config = LoRAConfig(**lora_config)

        # Create LoRA adapter
        lora_adapter = LoRAAdapter(base_model, lora_config)

        # Inject adapters if requested
        if inject_adapters:
            stats = lora_adapter.inject_adapters(verbose=verbose)
            if verbose:
                total_params = sum(p.size for p in base_model.parameters())
                lora_params = sum(stats.values())
                reduction = (1 - lora_params / total_params) * 100
                logger.info(f"Parameter reduction: {reduction:.1f}%")

        return base_model, lora_adapter

    def build_qlora_model(
        self,
        base_model: Module,
        qlora_config: QLoRAConfig | str | dict | None = None,
        quantize_base: bool = True,
        verbose: bool = True,
    ) -> tuple[Module, LoRAAdapter]:
        """Build a model with QLoRA (quantized base + LoRA adapters).
        
        Args:
            base_model: Base model to quantize and add LoRA to
            qlora_config: QLoRA configuration
            quantize_base: Whether to quantize the base model
            verbose: Print details
            
        Returns:
            Tuple of (model, lora_adapter)
        """
        logger.info("Building model with QLoRA")
        
        # Get QLoRA config
        if qlora_config is None:
            qlora_config = get_lora_preset("qlora_memory")
        elif isinstance(qlora_config, str):
            qlora_config = get_lora_preset(qlora_config)
        elif isinstance(qlora_config, dict):
            qlora_config = QLoRAConfig(**qlora_config)
            
        if not isinstance(qlora_config, QLoRAConfig):
            # Convert to QLoRA config if needed
            qlora_config = QLoRAConfig(**qlora_config.__dict__)

        # Quantize base model if requested
        if quantize_base:
            try:
                from ..quantization_utils import ModelQuantizer, QuantizationConfig

                quant_config = QuantizationConfig(
                    bits=4,
                    quantization_type=qlora_config.bnb_4bit_quant_type,
                    use_double_quant=qlora_config.bnb_4bit_use_double_quant,
                )

                quantizer = ModelQuantizer(quant_config)
                base_model = quantizer.quantize_model(base_model)
                logger.info("Quantized base model to 4-bit")
            except ImportError:
                logger.warning("Quantization utils not available, skipping quantization")

        # Create QLoRA adapter
        lora_adapter = LoRAAdapter(base_model, qlora_config)
        lora_adapter.inject_adapters(verbose=verbose)

        return base_model, lora_adapter

    def build_multi_adapter_model(
        self,
        base_model: Module,
        adapter_configs: dict[str, LoRAConfig | dict | str] | None = None,
    ) -> tuple[Module, MultiAdapterManager]:
        """Build a model with multiple LoRA adapters for multi-task learning.
        
        Args:
            base_model: Base model
            adapter_configs: Dict mapping adapter names to configs
            
        Returns:
            Tuple of (model, multi_adapter_manager)
        """
        logger.info("Building model with multiple LoRA adapters")
        
        # Create multi-adapter manager
        manager = MultiAdapterManager(base_model)

        # Add adapters
        if adapter_configs:
            for name, config in adapter_configs.items():
                # Process config
                if isinstance(config, str):
                    config = get_lora_preset(config)
                elif isinstance(config, dict):
                    if any(k.startswith("bnb_") for k in config):
                        config = QLoRAConfig(**config)
                    else:
                        config = LoRAConfig(**config)

                manager.add_adapter(name, config)
                logger.info(f"Added adapter '{name}'")

        return base_model, manager