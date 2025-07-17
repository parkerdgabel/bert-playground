# MLX Training Code Consolidation Plan

## Overview
This document outlines the comprehensive plan to consolidate our training infrastructure, merging the optimized MLX trainer with the MLflow-integrated trainer while ensuring a single, unified MLflow database.

## Current State Analysis

### Existing Components
1. **Trainers:**
   - `TitanicTrainerV2` - Full MLflow integration, extensive logging
   - `MLXOptimizedTrainer` - MLX-specific optimizations, memory management
   - Legacy trainers in various states

2. **Training Scripts:**
   - Multiple entry points: `train_titanic_v2.py`, `train_optimized.py`, `run_production.py`
   - Unified CLI: `mlx_bert_cli.py` (partially integrated)

3. **MLflow Databases:**
   - Fragmented across multiple locations
   - Different experiments in different databases
   - No central tracking location

## Consolidation Strategy

### Phase 1: Unified MLflow Configuration
**Goal:** Single MLflow database for all experiments

1. **Create Central MLflow Configuration:**
   ```python
   # utils/mlflow_central.py
   MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
   MLFLOW_ARTIFACT_LOCATION = "./mlruns/artifacts"
   ```

2. **Environment Configuration:**
   - Set `MLFLOW_TRACKING_URI` environment variable
   - Configure all components to use central database
   - Migrate existing experiments to central location

### Phase 2: Unified Trainer Architecture
**Goal:** Single trainer class with all optimizations and features

```
MLXTrainer (Base)
├── Core MLX Optimizations
│   ├── Lazy evaluation patterns
│   ├── Dynamic batch sizing
│   ├── Memory profiling
│   └── Gradient accumulation
├── MLflow Integration Layer
│   ├── Experiment tracking
│   ├── Metric logging
│   ├── Artifact management
│   └── Model registry
└── Advanced Features
    ├── Mixed precision training
    ├── Model quantization
    ├── Distributed training hooks
    └── Custom callbacks
```

### Phase 3: Implementation Plan

#### 1. Create Base MLXTrainer
```python
# training/mlx_trainer.py
class MLXTrainer:
    """Unified MLX trainer with all optimizations."""
    
    def __init__(self, config: TrainingConfig):
        # Core MLX optimizations from MLXOptimizedTrainer
        # MLflow integration from TitanicTrainerV2
        # Unified configuration system
```

#### 2. Configuration System
```python
# configs/training_config.py
@dataclass
class TrainingConfig:
    # Training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    
    # MLX optimizations
    base_batch_size: int = 32
    max_batch_size: int = 64
    enable_dynamic_batching: bool = True
    gradient_accumulation_steps: int = 1
    
    # MLflow settings
    enable_mlflow: bool = True
    experiment_name: str = "mlx_training"
    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    
    # Memory optimization
    enable_quantization: bool = False
    quantization_bits: int = 4
```

#### 3. Data Pipeline Consolidation
- Merge `unified_loader.py` and `optimized_loader.py`
- Keep MLX-Data optimizations
- Standardize preprocessing pipeline

### Phase 4: Migration Steps

1. **Week 1: Foundation**
   - [ ] Set up central MLflow configuration
   - [ ] Create base MLXTrainer class structure
   - [ ] Implement core MLX optimizations

2. **Week 2: Integration**
   - [ ] Add MLflow integration layer
   - [ ] Migrate features from TitanicTrainerV2
   - [ ] Consolidate data loading pipelines

3. **Week 3: Migration**
   - [ ] Update mlx_bert_cli.py to use new trainer
   - [ ] Create migration scripts for existing experiments
   - [ ] Test with existing workflows

4. **Week 4: Cleanup**
   - [ ] Remove legacy code
   - [ ] Update documentation
   - [ ] Final testing and validation

### Phase 5: Benefits

1. **Single Source of Truth**
   - One trainer implementation to maintain
   - Consistent behavior across all use cases
   - Easier debugging and development

2. **Unified MLflow Tracking**
   - All experiments in one database
   - Easy comparison across runs
   - Centralized model registry

3. **Optimized Performance**
   - All MLX optimizations in one place
   - Memory-efficient training
   - Production-ready out of the box

4. **Simplified Usage**
   ```bash
   # Single command for all training needs
   uv run python mlx_bert_cli.py train \
     --config configs/production.yaml \
     --experiment my_experiment
   ```

## File Structure After Consolidation

```
training/
├── mlx_trainer.py          # Unified trainer
├── callbacks.py            # Training callbacks
└── legacy/                 # Archived old trainers

configs/
├── training_config.py      # Unified configuration
└── presets/               # Pre-configured settings
    ├── quick.yaml
    ├── standard.yaml
    └── production.yaml

utils/
├── mlflow_central.py      # Central MLflow config
└── checkpoint_utils.py    # Unified checkpoint handling

mlruns/                    # Single MLflow location
├── mlflow.db             # Central database
└── artifacts/            # All artifacts
```

## Migration Checklist

- [ ] Backup existing MLflow databases
- [ ] Create central MLflow configuration
- [ ] Implement unified MLXTrainer
- [ ] Migrate existing training scripts
- [ ] Update CLI to use new trainer
- [ ] Consolidate data pipelines
- [ ] Standardize checkpoint formats
- [ ] Remove legacy code
- [ ] Update documentation
- [ ] Validate performance parity
- [ ] Create migration guide for users

## Success Criteria

1. **Performance:** No regression in training speed or memory usage
2. **Features:** All existing features preserved or improved
3. **Simplicity:** Reduced code duplication by >50%
4. **Tracking:** Single MLflow database with all experiments
5. **Usability:** Simpler API with fewer configuration options

## Next Steps

1. Review and approve this plan
2. Create feature branch for implementation
3. Begin Phase 1 implementation
4. Set up CI/CD for testing during migration