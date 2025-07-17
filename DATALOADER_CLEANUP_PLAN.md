# DataLoader Architecture Cleanup Plan

## Current State
We have multiple dataloader implementations with confusing naming:
- `dataloader.py` - Called "V2" but is actually our primary, efficient MLX-native implementation
- `universal_loader.py` - Complex, feature-rich but unused
- `v2_kaggle_adapter.py` - Temporary adapter to bridge old interfaces
- Various legacy references and imports

## Goal
Create a clean, efficient MLX-native dataset loader as the single source of truth for Kaggle problems.

## Proposed Architecture

### 1. Core DataLoader (`mlx_dataloader.py`)
Rename `dataloader.py` → `mlx_dataloader.py` and refine:
- **Class**: `KaggleDataLoader` (remove "Tabular" and "Text" - it's implicit)
- **Purpose**: Efficient MLX-native streaming for Kaggle tabular-to-text problems
- **Features**:
  - Stream-based data loading with MLX
  - Efficient preprocessing and tokenization
  - Built-in text template support
  - Caching capabilities
  - Multi-threaded prefetching

### 2. Text Generation (`text_generation.py`)
Consolidate text generation utilities:
- Move `TitanicTextTemplates` to a more generic implementation
- Support multiple Kaggle datasets out of the box
- Configurable templates per dataset

### 3. Dataset Registry (`datasets.py`)
Keep dataset specifications but simplify:
- Maintain `KaggleDatasetSpec` for metadata
- Pre-configured specs for common Kaggle competitions
- Easy registration of new datasets

## Files to Remove
1. `universal_loader.py` - Overly complex, unused
2. `mlx_streaming.py` - Only used by universal loader
3. `v2_kaggle_adapter.py` - No longer needed after cleanup
4. `enhanced_unified_loader.py` - Already deleted but still referenced
5. `unified_loader.py` - Already deleted but still referenced

## Files to Update
1. Rename `dataloader.py` → `mlx_dataloader.py`
2. Update class names to remove "V2" references
3. Simplify `__init__.py` exports
4. Fix broken imports in `trainer_v2.py`
5. Update all imports in CLI and tests

## Implementation Steps

### Phase 1: Core Refactoring
1. Create new `mlx_dataloader.py` with cleaned up implementation
2. Remove unused streaming utilities
3. Update text generation to be more generic

### Phase 2: Update Dependencies
1. Update CLI to use new dataloader
2. Fix trainer imports
3. Update all tests

### Phase 3: Cleanup
1. Remove old files
2. Update documentation
3. Remove "V2" references everywhere

## Benefits
- Single, clear dataloader implementation
- No confusing version numbers
- Efficient MLX-native design
- Easy to extend for new Kaggle competitions
- Clean, maintainable codebase