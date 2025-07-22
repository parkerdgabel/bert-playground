# Phase 2 Migration Guide

## Overview

Phase 2 of the k-bert project introduces a modern, event-driven architecture with hexagonal design patterns, enhanced plugin system, and improved training components. This guide provides a comprehensive overview of migrating from Phase 1 to Phase 2.

## What's New in Phase 2

### 1. Event-Driven Architecture
- Centralized event system for component communication
- Asynchronous training progress tracking
- Plugin lifecycle management through events
- Real-time metric collection and reporting

### 2. Hexagonal Architecture (Ports & Adapters)
- Clean separation between business logic and external concerns
- Testable, maintainable code structure
- Easy swapping of implementations
- Better dependency management

### 3. Enhanced Plugin System
- Type-safe plugin interfaces
- Dependency injection for plugins
- Plugin lifecycle management
- Hot-reloading capabilities

### 4. Modern Training Components
- Modular trainer architecture
- Composable training strategies
- Advanced callback system
- Distributed training support

### 5. Enhanced Data Pipeline
- Stream-based data processing
- Advanced preprocessing capabilities
- Efficient memory management
- Multi-format support

## Migration Timeline

### Phase 1: Preparation (Weeks 1-2)
1. **Assessment**: Audit existing configurations and custom components
2. **Planning**: Identify migration priorities and breaking changes
3. **Backup**: Create backups of existing projects and configurations
4. **Environment**: Set up Phase 2 development environment

### Phase 2: Core Migration (Weeks 3-4)
1. **Configuration**: Update configuration files to new schema
2. **Plugins**: Migrate custom plugins to new interfaces
3. **Training**: Update training scripts and configurations
4. **Testing**: Validate migrated components

### Phase 3: Optimization (Weeks 5-6)
1. **Event Integration**: Add event-driven features
2. **Architecture**: Refactor to hexagonal patterns
3. **Performance**: Optimize with new capabilities
4. **Documentation**: Update project documentation

## Key Breaking Changes

### Configuration Schema Changes
```yaml
# Phase 1 (Old)
model:
  type: "modernbert_with_head"
  params:
    num_labels: 2

# Phase 2 (New)
model:
  architecture: "modernbert"
  head:
    type: "classification"
    num_labels: 2
  adapter:
    type: "lora"
    rank: 16
```

### Plugin Interface Updates
```python
# Phase 1 (Old)
class CustomHead:
    def __init__(self, config):
        self.config = config
    
    def forward(self, x):
        return self.output(x)

# Phase 2 (New)
from bert_playground.core.ports import HeadPort
from bert_playground.core.events import EventBus

class CustomHead(HeadPort):
    def __init__(self, config: HeadConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
    
    async def forward(self, x: mx.array) -> mx.array:
        result = self.output(x)
        await self.event_bus.emit("head.forward_complete", 
                                {"head": self.__class__.__name__})
        return result
```

### Training Loop Changes
```python
# Phase 1 (Old)
trainer = Trainer(model, train_loader, val_loader)
trainer.train()

# Phase 2 (New)
from bert_playground.training.factory import TrainerFactory
from bert_playground.core.events import EventBus

event_bus = EventBus()
trainer = TrainerFactory.create(
    config=trainer_config,
    event_bus=event_bus
)
await trainer.train_async()
```

## Migration Strategies

### 1. Gradual Migration (Recommended)
- Migrate one component at a time
- Run Phase 1 and Phase 2 in parallel
- Validate each migration step
- Use feature flags for gradual rollout

### 2. Big Bang Migration
- Complete migration in one go
- Requires thorough testing
- Higher risk but faster completion
- Recommended for smaller projects

### 3. Hybrid Approach
- Keep critical components in Phase 1
- Migrate non-critical components first
- Gradually move critical components
- Maintain backward compatibility

## Step-by-Step Migration Process

### Step 1: Environment Setup
```bash
# Update to latest k-bert version
uv sync

# Install Phase 2 dependencies
uv add k-bert[phase2]

# Initialize Phase 2 configuration
k-bert config init --phase2
```

### Step 2: Configuration Migration
```bash
# Migrate configuration files
k-bert migrate config --input k-bert.yaml --output k-bert-phase2.yaml

# Validate new configuration
k-bert config validate k-bert-phase2.yaml
```

### Step 3: Plugin Migration
```bash
# Generate plugin templates
k-bert plugin migrate --input src/plugins --output src/plugins-phase2

# Update plugin interfaces
k-bert plugin update-interfaces src/plugins-phase2
```

### Step 4: Training Migration
```bash
# Migrate training scripts
k-bert training migrate --input training/ --output training-phase2/

# Test new training pipeline
k-bert train --config k-bert-phase2.yaml --dry-run
```

### Step 5: Validation
```bash
# Run comprehensive tests
uv run pytest tests/migration/

# Performance comparison
k-bert benchmark --phase1-config k-bert.yaml --phase2-config k-bert-phase2.yaml
```

## Common Migration Issues

### 1. Configuration Validation Errors
**Issue**: Old configuration schema not compatible
**Solution**: Use migration tool or update manually following new schema

### 2. Plugin Interface Mismatches
**Issue**: Plugins using old interfaces
**Solution**: Update plugin base classes and method signatures

### 3. Async/Await Requirements
**Issue**: Synchronous code in async context
**Solution**: Add async/await keywords or use sync wrappers

### 4. Event System Integration
**Issue**: Components not receiving events
**Solution**: Ensure proper event bus wiring and subscription

### 5. Dependency Injection Issues
**Issue**: Components not properly injected
**Solution**: Update container configuration and component factories

## Testing Migration

### Unit Tests
```python
# Test old and new implementations side by side
def test_head_compatibility():
    old_head = OldCustomHead(config)
    new_head = NewCustomHead(config, event_bus)
    
    x = mx.random.normal((32, 768))
    
    old_result = old_head.forward(x)
    new_result = asyncio.run(new_head.forward(x))
    
    assert mx.allclose(old_result, new_result)
```

### Integration Tests
```python
# Test complete pipeline migration
async def test_training_pipeline_migration():
    # Setup both pipelines
    old_trainer = create_old_trainer()
    new_trainer = await create_new_trainer()
    
    # Compare training results
    old_metrics = old_trainer.train()
    new_metrics = await new_trainer.train_async()
    
    assert_metrics_similar(old_metrics, new_metrics)
```

### Performance Tests
```python
# Compare performance between phases
def test_performance_regression():
    with Timer() as old_timer:
        run_phase1_training()
    
    with Timer() as new_timer:
        asyncio.run(run_phase2_training())
    
    assert new_timer.elapsed <= old_timer.elapsed * 1.1  # Allow 10% overhead
```

## Rollback Strategy

### Immediate Rollback
```bash
# Switch back to Phase 1 configuration
cp k-bert-backup.yaml k-bert.yaml

# Use Phase 1 entry points
k-bert --phase1 train

# Restore old plugins
git checkout HEAD~1 src/plugins/
```

### Gradual Rollback
```bash
# Disable specific Phase 2 features
k-bert config set features.event_system false
k-bert config set features.hexagonal false

# Use compatibility mode
k-bert --compatibility-mode train
```

## Success Metrics

### Technical Metrics
- **Performance**: Training time within 5% of Phase 1
- **Memory**: Memory usage not increased by more than 10%
- **Accuracy**: Model accuracy maintained or improved
- **Reliability**: No increase in error rates

### Operational Metrics
- **Migration Time**: Total migration time under planned duration
- **Downtime**: Minimal service interruption
- **Test Coverage**: Maintained or improved test coverage
- **Documentation**: Updated documentation for all changes

## Next Steps

After successful migration:

1. **Explore New Features**: Leverage event system and hexagonal architecture
2. **Optimize Performance**: Use advanced Phase 2 capabilities
3. **Enhance Monitoring**: Implement comprehensive event-based monitoring
4. **Scale Architecture**: Design for larger, more complex projects
5. **Contribute Back**: Share improvements with the community

## Support and Resources

- **Migration Tool**: `k-bert migrate --help`
- **Documentation**: See individual component guides
- **Examples**: Check `examples/phase2/migration_example/`
- **Community**: Join discussions on GitHub
- **Support**: Open issues for migration problems

## Conclusion

Phase 2 migration brings significant architectural improvements while maintaining backward compatibility where possible. Following this guide ensures a smooth transition to the enhanced k-bert platform.

The investment in migration pays off through improved maintainability, testability, and extensibility of your ML training infrastructure.