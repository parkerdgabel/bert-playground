# Web UI Adapters

This directory is prepared for future web UI implementation.

## Planned Structure

```
web/
├── app.py              # Web application (Flask/Django/FastAPI)
├── templates/          # HTML templates
│   ├── base.html      # Base layout
│   ├── train.html     # Training interface
│   └── predict.html   # Prediction interface
├── static/            # Static assets
│   ├── css/          # Stylesheets
│   ├── js/           # JavaScript
│   └── img/          # Images
├── views/            # View handlers
│   ├── training.py   # Training views
│   ├── prediction.py # Prediction views
│   └── dashboard.py  # Dashboard views
└── websocket/        # WebSocket handlers
    └── progress.py   # Real-time progress updates
```

## Design Principles

The web adapters will:

1. **Server-Side Rendering**: Initial approach for simplicity
2. **Progressive Enhancement**: Add interactivity with JavaScript
3. **Real-Time Updates**: WebSocket support for training progress
4. **Responsive Design**: Mobile-friendly interface
5. **DTO-Based**: Same DTOs as CLI and API

## Example View

```python
@app.route("/train", methods=["GET", "POST"])
async def train_view():
    if request.method == "POST":
        # 1. Parse form data
        form_data = request.form
        
        # 2. Create DTO
        dto = TrainingRequestDTO(...)
        
        # 3. Call use case
        use_case = get_service(TrainModelUseCase)
        response = await use_case.execute(dto)
        
        # 4. Render result
        return render_template("train_result.html", result=response)
    
    return render_template("train_form.html")
```

## Future Enhancements

- **SPA Support**: React/Vue/Svelte frontend
- **GraphQL**: Alternative to REST API
- **Real-time Monitoring**: Live charts and metrics
- **Multi-user Support**: User management and isolation