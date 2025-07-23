# CLI Migration Example

This document shows how the CLI was transformed from containing business logic to being a thin adapter.

## Before: CLI with Business Logic

```python
# cli/commands/core/train.py (OLD)
def train_command(...):
    """Train command with embedded business logic."""
    
    # Configuration loading logic
    config_provider = get_service(ConfigurationProvider)
    if config:
        config_provider.load_file(str(config))
    # ... more config logic ...
    
    # Model creation logic
    model_factory = get_service(ModelFactory)
    model = model_factory.create_model(
        model_name=model_name,
        model_type="modernbert_with_head",
        head_type="binary_classification",
        num_labels=2
    )
    
    # Data loading logic
    dataset_factory = get_service(DatasetFactory)
    train_loader = dataset_factory.create_dataloader(
        data_path=train_path,
        batch_size=batch_size,
        shuffle=True,
        split="train"
    )
    
    # Training orchestration logic
    orchestrator = get_service(TrainingOrchestrator)
    orchestrator.configure(training_config)
    result = orchestrator.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # ... lots more business logic ...
```

## After: Thin CLI Adapter

```python
# adapters/primary/cli/train_adapter.py (NEW)
async def train_command(...):
    """Thin adapter - only handles CLI concerns."""
    
    try:
        # 1. Parse arguments and create DTO
        request = create_train_request_dto(
            config_provider=get_service(ConfigurationProvider),
            config=config,
            experiment=experiment,
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            output_dir=output_dir,
            no_config=no_config,
        )
        
        # 2. Display configuration (UI concern)
        display_configuration_summary(request)
        
        # 3. Call use case (single line!)
        use_case = get_service(TrainModelUseCase)
        response = await use_case.execute(request)
        
        # 4. Display results (UI concern)
        display_training_results(response)
        
    except Exception as e:
        # Error formatting for CLI
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
```

## Key Differences

### 1. **Separation of Concerns**
- **Old**: CLI mixed UI, configuration, model creation, data loading, and training
- **New**: CLI only handles argument parsing, DTO creation, and display

### 2. **Business Logic Location**
- **Old**: Scattered throughout CLI command
- **New**: Centralized in `application/use_cases/train_model.py`

### 3. **Testing**
- **Old**: Hard to test without mocking many dependencies
- **New**: Easy to test - just verify DTO creation and display logic

### 4. **Reusability**
- **Old**: Logic tied to CLI, can't reuse for API or Web
- **New**: Use case can be called from CLI, API, or Web adapters

### 5. **Maintainability**
- **Old**: Changes to training logic require modifying CLI
- **New**: CLI remains stable even when training logic changes

## Benefits

1. **Single Responsibility**: Each component has one clear purpose
2. **Testability**: Thin adapters are easy to test
3. **Flexibility**: Same use cases work for CLI, API, and Web
4. **Maintainability**: Changes are isolated to appropriate layers
5. **Clarity**: Clear data flow through DTOs

## Migration Pattern

To migrate a CLI command:

1. Identify all business logic in the command
2. Move business logic to a use case in `application/use_cases/`
3. Create DTOs in `application/dto/` for request/response
4. Create thin adapter that only:
   - Parses arguments
   - Creates request DTO
   - Calls use case
   - Displays response
5. Remove the old command file

This pattern ensures the CLI remains a thin, stable interface while all business logic is properly organized in the application layer.