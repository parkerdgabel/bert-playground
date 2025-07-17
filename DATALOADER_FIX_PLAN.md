# MLX Data Loader Fix Plan

## Problem Analysis

The current dataloader implementation has a critical issue with MLX data streaming:

### Root Cause
1. **String Data Issue**: MLX data's `buffer_from_vector()` cannot handle dictionaries containing string values
2. **Order of Operations**: The current flow tries to create a stream from raw CSV data (with strings) before tokenization
3. **Type Mismatch**: MLX expects numeric arrays or bytes, but we're passing Python dicts with mixed types

### Error Flow
```
CSV → Pandas DataFrame → Dict Records (with strings) → dx.buffer_from_vector() → ERROR
                                                          ↑
                                                    Expects numeric data
```

## Proposed Solution

### Option 1: Pre-tokenization Pipeline (Recommended)
Transform data BEFORE creating the MLX stream:

```
CSV → Pandas → Text Generation → Tokenization → Numeric Arrays → MLX Stream
```

**Advantages:**
- Clean separation of concerns
- Easier to debug
- Can cache tokenized data
- Better performance for multiple epochs

**Implementation Steps:**
1. Load CSV with pandas
2. Generate text from tabular data
3. Tokenize all text
4. Convert to numeric arrays
5. Create MLX stream from numeric data

### Option 2: Custom Stream Transformation
Use MLX data's transformation capabilities:

```
CSV → Basic Stream → Transform (text generation + tokenization) → Batching
```

**Advantages:**
- More "streaming" oriented
- Potentially lower memory usage
- Follows MLX data patterns

**Disadvantages:**
- More complex to implement
- Harder to debug
- May have performance overhead

### Option 3: Direct Numeric Loading
Skip text generation for numeric features:

```
CSV → Extract Numeric Features → MLX Stream
     → Extract Text Features → Tokenize → Merge
```

**Advantages:**
- Can leverage MLX's CSV reader for numeric data
- Flexible handling of different data types

**Disadvantages:**
- More complex merging logic
- May not utilize text features effectively

## Implementation Plan

### Phase 1: Quick Fix (Option 1 - Pre-tokenization)
1. **Modify `_create_pandas_stream()` in `mlx_streaming.py`**:
   - Load CSV with pandas
   - Apply text transformation to create text from tabular data
   - Tokenize all samples
   - Convert to numpy arrays
   - Create stream from numeric arrays

2. **Update data flow**:
   ```python
   def _create_pandas_stream(self):
       # Load CSV
       df = pd.read_csv(self.csv_path)
       
       # Generate text for each row
       texts = [self.text_transform_fn(row) for _, row in df.iterrows()]
       
       # Tokenize all texts
       tokens = self.tokenizer(texts, padding="max_length", truncation=True, 
                              max_length=self.config.max_length, return_tensors="np")
       
       # Create numeric records
       records = []
       for i in range(len(df)):
           record = {
               "input_ids": tokens["input_ids"][i],
               "attention_mask": tokens["attention_mask"][i],
               "labels": int(df.iloc[i][self.dataset_spec.target_column])
           }
           records.append(record)
       
       # Create stream from numeric data
       return dx.buffer_from_vector(records)
   ```

### Phase 2: Optimization
1. **Add caching**:
   - Cache tokenized data to disk
   - Load from cache on subsequent runs

2. **Memory optimization**:
   - Process in chunks for large datasets
   - Use generators where possible

3. **Performance tuning**:
   - Parallel tokenization
   - Optimized text generation

### Phase 3: Advanced Features
1. **Dynamic batching by sequence length**
2. **On-the-fly augmentation**
3. **Multi-dataset support**

## Testing Strategy

1. **Unit Tests**:
   - Test tokenization pipeline
   - Test stream creation with numeric data
   - Test batch generation

2. **Integration Tests**:
   - End-to-end data loading
   - Training loop compatibility
   - Memory usage monitoring

3. **Performance Tests**:
   - Throughput measurement
   - Memory profiling
   - Comparison with original implementation

## Risks and Mitigations

1. **Memory Usage**: Pre-tokenizing entire dataset
   - Mitigation: Implement chunked processing

2. **Performance**: Tokenization overhead
   - Mitigation: Add caching layer

3. **Compatibility**: Breaking changes to data pipeline
   - Mitigation: Create new implementation alongside old one

## Timeline

- **Day 1**: Implement Phase 1 quick fix
- **Day 2**: Test and debug
- **Day 3**: Add basic optimizations
- **Day 4**: Performance testing
- **Day 5**: Documentation and cleanup

## Success Criteria

1. ✅ MLX data streaming works without errors
2. ✅ Can complete a full training epoch
3. ✅ Performance is acceptable (>100 samples/sec)
4. ✅ Memory usage is reasonable (<8GB for Titanic dataset)
5. ✅ All tests pass