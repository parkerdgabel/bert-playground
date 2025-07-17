# MLX DataLoader Implementation Plan V2 - Idiomatic Approach

## Key MLX Data Concepts

Based on the research, MLX-data has specific patterns and best practices:

### 1. **Core Abstractions**
- **Buffer**: Random-access data structure for in-memory datasets
- **Stream**: Sequential data structure for file-based or infinite datasets
- **Sample**: Dictionary of arrays (all samples are dict[str, array])

### 2. **Key Principles**
- **Lazy Evaluation**: Operations build a computation graph, executed only when needed
- **Unified Memory**: No CPU/GPU copying - data lives in shared memory
- **Multi-threaded Processing**: Avoid Python GIL by using built-in transformations
- **Framework Agnostic**: Can work with PyTorch, JAX, or MLX

## Idiomatic MLX Data Loading Pattern

### The MLX Way:
```
CSV → Stream → Transform (text generation) → Tokenize (CharTrie) → Batch → MLX Arrays
```

## Revised Implementation Plan

### Phase 1: Stream-Based Loading with Built-in Tokenization

#### 1.1 Use MLX's CSV Stream Reader
```python
import mlx.data as dx

# Create stream directly from CSV
stream = dx.stream_csv_reader(
    file="data/titanic/train.csv",
    sep=",",
    quote='"'
)
```

#### 1.2 Transform CSV Records to Text
```python
def record_to_text(sample):
    """Transform tabular record to text - runs in parallel."""
    # This is a pure function that MLX can parallelize
    text_parts = []
    
    if sample.get('Pclass'):
        text_parts.append(f"Passenger class: {sample['Pclass']}")
    if sample.get('Name'):
        text_parts.append(f"Name: {sample['Name']}")
    # ... other fields
    
    sample['text'] = " ".join(text_parts)
    return sample

# Apply transformation - this is lazy and parallel
stream = stream.sample_transform(record_to_text)
```

#### 1.3 Use MLX CharTrie for Tokenization
```python
# Build CharTrie from tokenizer vocabulary
from transformers import AutoTokenizer
import mlx.data.core as dx_core

# Load HuggingFace tokenizer just for vocabulary
hf_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# Create CharTrie from vocabulary
vocab = hf_tokenizer.get_vocab()
trie = dx_core.CharTrie()
for token, token_id in vocab.items():
    trie.insert(token.encode('utf-8'), token_id)

# Tokenize using MLX's built-in tokenization
stream = stream.tokenize(
    key='text',
    trie=trie,
    output_key='input_ids'
)
```

#### 1.4 Handle Padding and Attention Masks
```python
# Add padding transformation
def add_padding_and_mask(sample, max_length=128):
    """Add padding and attention mask."""
    input_ids = sample['input_ids']
    
    # Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    
    # Pad
    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        input_ids = np.concatenate([input_ids, np.zeros(pad_length, dtype=np.int32)])
    
    # Create attention mask
    attention_mask = np.ones_like(input_ids)
    if pad_length > 0:
        attention_mask[-pad_length:] = 0
    
    sample['input_ids'] = input_ids
    sample['attention_mask'] = attention_mask
    
    # Add label
    sample['labels'] = int(sample.get('Survived', 0))
    
    # Remove text key to save memory
    sample.pop('text', None)
    
    return sample

stream = stream.sample_transform(add_padding_and_mask)
```

#### 1.5 Batching and Prefetching
```python
# Apply MLX optimizations
stream = stream.shuffle(buffer_size=1000)  # Shuffle for training
stream = stream.batch(batch_size=32)       # Batch samples
stream = stream.prefetch(prefetch_size=4)  # Prefetch for performance

# Convert to MLX arrays in batch
def batch_to_mlx(batch):
    """Convert numpy arrays to MLX arrays."""
    return {
        'input_ids': mx.array(np.stack(batch['input_ids'])),
        'attention_mask': mx.array(np.stack(batch['attention_mask'])),
        'labels': mx.array(batch['labels'])
    }

stream = stream.batch_transform(batch_to_mlx)
```

### Phase 2: Optimized Implementation

#### 2.1 Create Reusable DataLoader Class
```python
class MLXTabularTextDataLoader:
    """Idiomatic MLX data loader for tabular-to-text tasks."""
    
    def __init__(self, csv_path, tokenizer_name, max_length=128, batch_size=32):
        self.csv_path = csv_path
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Build CharTrie from tokenizer
        self.trie = self._build_trie(tokenizer_name)
        
    def _build_trie(self, tokenizer_name):
        """Build CharTrie from HuggingFace tokenizer."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        trie = dx_core.CharTrie()
        for token, token_id in tokenizer.get_vocab().items():
            # Handle special tokens
            token_bytes = token.encode('utf-8', errors='ignore')
            if token_bytes:
                trie.insert(token_bytes, token_id)
        
        return trie
    
    def create_stream(self, is_training=True):
        """Create MLX data stream."""
        # Start with CSV reader
        stream = dx.stream_csv_reader(self.csv_path)
        
        # Transform to text
        stream = stream.sample_transform(self._record_to_text)
        
        # Tokenize with CharTrie
        stream = stream.tokenize('text', self.trie, output_key='input_ids')
        
        # Add padding and masks
        stream = stream.sample_transform(
            lambda s: self._add_padding_and_mask(s, self.max_length)
        )
        
        # Training optimizations
        if is_training:
            stream = stream.shuffle(buffer_size=1000)
        
        # Batch and prefetch
        stream = stream.batch(self.batch_size)
        stream = stream.prefetch(4)
        
        # Convert to MLX arrays
        stream = stream.batch_transform(self._batch_to_mlx)
        
        return stream
```

#### 2.2 Alternative: Buffer-Based for Small Datasets
```python
def create_buffer_pipeline(csv_path, tokenizer_name, max_length=128):
    """For smaller datasets that fit in memory."""
    
    # Load entire dataset
    df = pd.read_csv(csv_path)
    
    # Convert to list of dicts
    records = df.to_dict('records')
    
    # Create buffer
    buffer = dx.buffer_from_vector(records)
    
    # Apply transformations
    buffer = buffer.sample_transform(record_to_text)
    buffer = buffer.tokenize('text', trie, output_key='input_ids')
    buffer = buffer.sample_transform(lambda s: add_padding_and_mask(s, max_length))
    
    # Shuffle entire buffer
    buffer = buffer.shuffle()
    
    return buffer
```

### Phase 3: Performance Optimizations

#### 3.1 Caching Tokenized Data
```python
def create_cached_stream(csv_path, cache_path, tokenizer_name):
    """Cache tokenized data for faster subsequent loads."""
    
    if Path(cache_path).exists():
        # Load from cache
        return dx.stream_csv_reader(cache_path)
    
    # Create and cache
    stream = create_stream(csv_path, tokenizer_name)
    
    # Materialize to cache
    cached_samples = []
    for batch in stream.take(float('inf')):
        cached_samples.extend(batch)
    
    # Save cache
    pd.DataFrame(cached_samples).to_csv(cache_path, index=False)
    
    return dx.stream_csv_reader(cache_path)
```

#### 3.2 Multi-threaded Text Generation
```python
# Use C++ extensions for text generation if needed
def register_text_transform():
    """Register C++ text transformation for performance."""
    # This would be implemented in C++ for maximum performance
    # For now, use Python with minimal GIL impact
    pass
```

## Key Differences from Original Plan

1. **Stream-First**: Use `stream_csv_reader` directly instead of pandas loading
2. **CharTrie Tokenization**: Use MLX's built-in tokenizer instead of HuggingFace
3. **Lazy Transformations**: All operations are lazy until iteration
4. **No String Arrays**: Transform strings to tokens before creating arrays
5. **Unified Memory**: No explicit device management needed

## Testing Strategy

```python
# Test basic streaming
stream = dx.stream_csv_reader("data/titanic/train.csv")
sample = next(iter(stream))
assert isinstance(sample, dict)

# Test tokenization
stream = stream.tokenize('text', trie)
sample = next(iter(stream))
assert 'input_ids' in sample

# Test batching
stream = stream.batch(32)
batch = next(iter(stream))
assert len(batch['input_ids']) == 32
```

## Benefits of This Approach

1. **True Streaming**: Data is processed on-demand, not loaded entirely
2. **Parallel Processing**: Transformations run on multiple threads
3. **Memory Efficient**: Only active batches in memory
4. **MLX Native**: Uses MLX's optimized operations
5. **Framework Agnostic**: Can be used with PyTorch or JAX too

## Implementation Timeline

- **Day 1**: Implement basic streaming pipeline
- **Day 2**: Add CharTrie tokenization
- **Day 3**: Optimize and test performance
- **Day 4**: Add caching and advanced features
- **Day 5**: Integration testing with trainer

This approach follows MLX's design philosophy and should provide much better performance than the original pandas-based approach.