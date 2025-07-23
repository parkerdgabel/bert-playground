# REST API Adapters

This directory is prepared for future REST API implementation.

## Planned Structure

```
api/
├── app.py              # FastAPI/Flask application
├── routers/            # API route handlers
│   ├── training.py     # Training endpoints
│   ├── prediction.py   # Prediction endpoints
│   └── models.py       # Model management endpoints
├── middleware/         # API middleware
│   ├── auth.py        # Authentication
│   ├── cors.py        # CORS handling
│   └── logging.py     # Request logging
└── schemas/           # Pydantic schemas for API
    ├── requests.py    # Request schemas
    └── responses.py   # Response schemas
```

## Design Principles

The API adapters will follow the same principles as CLI adapters:

1. **Thin Layer**: Only handle HTTP concerns (parsing, validation, responses)
2. **DTO-Based**: Convert between HTTP requests/responses and application DTOs
3. **No Business Logic**: All logic resides in use cases
4. **Async First**: Leverage async/await for performance

## Example Endpoint

```python
@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    # 1. Convert API request to DTO
    dto = TrainingRequestDTO(...)
    
    # 2. Call use case
    use_case = get_service(TrainModelUseCase)
    response = await use_case.execute(dto)
    
    # 3. Convert DTO to API response
    return TrainingResponse.from_dto(response)
```