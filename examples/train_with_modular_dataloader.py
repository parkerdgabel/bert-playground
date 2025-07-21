"""
Example of using the modular dataloader system for training.
Shows how to integrate text conversion, configuration, and caching.
"""

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from embeddings.tokenizer_wrapper import TokenizerWrapper

from data.configs import ConfigFactory
from data.text_conversion import TextConverterFactory
from data.utils.caching import DiskCache, TokenizationCache

console = Console()


def create_titanic_dataloader(config_path: str = None):
    """
    Create a dataloader for Titanic competition using the modular system.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured dataloader components
    """
    console.print("[bold blue]Creating Titanic DataLoader[/bold blue]\n")

    # 1. Create configuration
    if config_path:
        config = ConfigFactory.from_file(config_path)
    else:
        # Use competition preset
        config = ConfigFactory.create_competition_config(
            "titanic",
            data_path="data/titanic/train.csv",
            batch_size=32,
            max_length=256,
            augment=True,
            optimization_profile="balanced",
        )

    console.print("[yellow]Configuration:[/yellow]")
    console.print(f"  Competition: {config.competition_name}")
    console.print(f"  Batch size: {config.dataloader.batch_size}")
    console.print(f"  Max length: {config.dataloader.max_length}")
    console.print(f"  Workers: {config.dataloader.num_workers}")
    console.print(f"  Cache enabled: {config.dataloader.enable_cache}\n")

    # 2. Create text converter
    converter = TextConverterFactory.create(
        config.dataloader.text_converter,
        augment=config.dataloader.augment,
    )

    # 3. Create caches if enabled
    caches = {}
    if config.dataloader.enable_cache:
        # Text conversion cache
        if config.dataloader.cache_converted_text:
            caches["text"] = DiskCache(
                cache_dir=Path(config.dataloader.cache_dir or "./cache") / "text",
                max_size_mb=500,
            )

        # Tokenization cache
        if config.dataloader.cache_tokenized:
            caches["tokenization"] = TokenizationCache(
                cache_dir=Path(config.dataloader.cache_dir or "./cache")
                / "tokenization",
                tokenizer_name=config.dataloader.tokenizer_name,
            )

    # 4. Create tokenizer
    tokenizer = TokenizerWrapper(
        model_name=config.dataloader.tokenizer_name,
        backend=config.dataloader.tokenizer_backend,
    )

    # 5. Load and prepare data
    df = pd.read_csv(config.dataset.data_path)
    df.columns = df.columns.str.lower()  # Normalize column names

    console.print("[yellow]Dataset:[/yellow]")
    console.print(f"  Total samples: {len(df)}")
    console.print(f"  Features: {list(df.columns)[:5]}...")
    console.print(f"  Label column: {config.dataset.label_column}\n")

    return {
        "config": config,
        "converter": converter,
        "tokenizer": tokenizer,
        "caches": caches,
        "data": df,
    }


def process_batch(batch_data, converter, tokenizer, caches):
    """
    Process a batch of data through the pipeline.

    Args:
        batch_data: List of data dictionaries
        converter: Text converter
        tokenizer: Tokenizer wrapper
        caches: Dictionary of caches

    Returns:
        Processed batch
    """
    batch_texts = []
    batch_labels = []

    for sample in batch_data:
        # Check text cache
        sample_id = str(sample.get("passengerid", id(sample)))

        if "text" in caches:
            cached_text = caches["text"].get(sample_id)
            if cached_text:
                text = cached_text
            else:
                # Convert to text
                result = converter(sample)
                text = result["text"]
                caches["text"].set(sample_id, text)
        else:
            # Convert without cache
            result = converter(sample)
            text = result["text"]

        batch_texts.append(text)
        batch_labels.append(sample.get("survived", 0))

    # Tokenize batch
    if "tokenization" in caches:
        # Check tokenization cache for each text
        encodings = []
        for text in batch_texts:
            cached = caches["tokenization"].get_tokenized(
                text=text,
                max_length=256,  # Use config value
                padding="max_length",
                truncation=True,
            )
            if cached is not None:
                encodings.append(cached)
            else:
                # Tokenize
                # Use batch_encode_plus for single text to get full output
                encoding = tokenizer.batch_encode_plus(
                    [text],  # Single text as list
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )
                # Extract single sample from batch and convert to lists for caching
                encoding_dict = {
                    k: v[0].tolist() if hasattr(v[0], "tolist") else v[0]
                    for k, v in encoding.items()
                }
                # Cache it
                caches["tokenization"].set_tokenized(
                    text=text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    encoding=encoding_dict,
                )
                # Keep MLX arrays for immediate use
                encoding = {k: v[0] for k, v in encoding.items()}
                encodings.append(encoding)
    else:
        # Tokenize without cache
        # Tokenize all at once for efficiency
        batch_encoding = tokenizer.batch_encode_plus(
            batch_texts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        # Convert to list of dicts
        encodings = [
            {k: v[i] for k, v in batch_encoding.items()}
            for i in range(len(batch_texts))
        ]

    # Create batch arrays
    import mlx.core as mx

    # Stack arrays from encodings
    # The encodings can be either dicts with MLX arrays or dicts with lists (from cache)
    if encodings and isinstance(encodings[0], dict):
        first_val = encodings[0].get("input_ids")
        if isinstance(first_val, mx.array):
            # Already MLX arrays
            # Ensure all arrays have the same shape before stacking
            input_ids_list = []
            attention_mask_list = []
            for e in encodings:
                # Get the arrays
                ids = e["input_ids"]
                mask = e["attention_mask"]
                # Ensure they're 1D (not 0D or 2D)
                if ids.ndim == 0:
                    ids = ids.reshape(1)
                elif ids.ndim > 1:
                    ids = ids.squeeze()
                if mask.ndim == 0:
                    mask = mask.reshape(1)
                elif mask.ndim > 1:
                    mask = mask.squeeze()
                input_ids_list.append(ids)
                attention_mask_list.append(mask)

            input_ids = mx.stack(input_ids_list)
            attention_mask = mx.stack(attention_mask_list).astype(mx.float32)
        else:
            # Convert from lists (cached data)
            input_ids = mx.array([e["input_ids"] for e in encodings])
            attention_mask = mx.array(
                [e["attention_mask"] for e in encodings], dtype=mx.float32
            )
    else:
        raise ValueError(
            f"Unexpected encoding format: {type(encodings[0]) if encodings else 'empty'}"
        )

    labels = mx.array(batch_labels, dtype=mx.int32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "texts": batch_texts,  # Keep for debugging
    }


def main():
    """Run the example."""
    console.print("[bold green]Modular DataLoader Training Example[/bold green]\n")

    # Create dataloader components
    components = create_titanic_dataloader()

    # Process a sample batch
    console.print("[bold blue]Processing Sample Batch[/bold blue]\n")

    # Get batch data
    batch_size = components["config"].dataloader.batch_size
    batch_data = components["data"].head(batch_size).to_dict("records")

    # Process batch
    batch = process_batch(
        batch_data,
        components["converter"],
        components["tokenizer"],
        components["caches"],
    )

    console.print("[yellow]Batch shapes:[/yellow]")
    console.print(f"  Input IDs: {batch['input_ids'].shape}")
    console.print(f"  Attention mask: {batch['attention_mask'].shape}")
    console.print(f"  Labels: {batch['labels'].shape}\n")

    console.print("[yellow]Sample texts:[/yellow]")
    for i in range(min(3, len(batch["texts"]))):
        console.print(f"  {i + 1}. {batch['texts'][i][:100]}...")

    console.print(f"\n[yellow]Labels:[/yellow] {batch['labels'][:10].tolist()}...")

    # Show cache statistics
    if components["caches"]:
        console.print("\n[yellow]Cache statistics:[/yellow]")
        for name, cache in components["caches"].items():
            if hasattr(cache, "cache"):
                console.print(f"  {name}: {len(cache.cache)} items cached")

    console.print("\n[green]✓ DataLoader ready for training![/green]")
    console.print("\nNext steps:")
    console.print("1. Create MLX data stream from the processed batches")
    console.print("2. Initialize your BERT model")
    console.print("3. Start training with the modular dataloader")

    # Save example config
    config_path = Path("configs/titanic_example.json")
    config_path.parent.mkdir(exist_ok=True)
    components["config"].save(config_path)
    console.print(f"\n[dim]Saved example config to {config_path}[/dim]")


if __name__ == "__main__":
    main()
