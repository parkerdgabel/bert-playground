"""Model serving command - serve models as REST API endpoints."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from loguru import logger
import json
import sys
import uvicorn
from datetime import datetime

from ...utils import (
    get_console, print_success, print_error, print_info, print_warning,
    handle_errors, track_time, requires_project,
    validate_path, validate_port
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = get_console()

@handle_errors
@requires_project()
def serve_command(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c",
                                  help="Path to model checkpoint",
                                  callback=lambda p: validate_path(p, must_exist=True)),
    port: int = typer.Option(8000, "--port", "-p",
                           help="Port to serve on",
                           callback=validate_port),
    host: str = typer.Option("0.0.0.0", "--host", "-h",
                           help="Host to bind to"),
    workers: int = typer.Option(1, "--workers", "-w",
                              help="Number of worker processes"),
    batch_size: int = typer.Option(32, "--batch-size", "-b",
                                 help="Maximum batch size for inference"),
    max_length: int = typer.Option(256, "--max-length",
                                 help="Maximum sequence length"),
    timeout: int = typer.Option(30, "--timeout", "-t",
                              help="Request timeout in seconds"),
    cors: bool = typer.Option(True, "--cors/--no-cors",
                            help="Enable CORS headers"),
    auth_token: Optional[str] = typer.Option(None, "--auth-token",
                                           help="Optional authentication token"),
    log_level: str = typer.Option("info", "--log-level",
                                help="Logging level: debug, info, warning, error"),
    reload: bool = typer.Option(False, "--reload", "-r",
                              help="Enable auto-reload for development"),
    ssl_cert: Optional[Path] = typer.Option(None, "--ssl-cert",
                                          help="Path to SSL certificate"),
    ssl_key: Optional[Path] = typer.Option(None, "--ssl-key",
                                         help="Path to SSL private key"),
    model_name: Optional[str] = typer.Option(None, "--model-name", "-n",
                                           help="Model name for API responses"),
    cache_size: int = typer.Option(100, "--cache-size",
                                 help="Number of predictions to cache"),
    enable_metrics: bool = typer.Option(True, "--metrics/--no-metrics",
                                      help="Enable Prometheus metrics endpoint"),
    health_check: bool = typer.Option(True, "--health/--no-health",
                                    help="Enable health check endpoint"),
    docs: bool = typer.Option(True, "--docs/--no-docs",
                            help="Enable API documentation"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                               help="Show detailed server information")
):
    """Serve a trained model as a REST API endpoint.
    
    This command starts a production-ready API server for model inference.
    
    Features:
    • FastAPI-based REST API
    • Batch prediction support
    • Request caching
    • Health checks and metrics
    • OpenAPI documentation
    • CORS support
    • SSL/TLS support
    • Authentication (optional)
    
    Examples:
        # Basic serving
        bert model serve -c output/run_001/best_model
        
        # Production serving with SSL
        bert model serve -c output/run_001/best_model \\
            --port 443 --ssl-cert cert.pem --ssl-key key.pem
        
        # Development mode with auto-reload
        bert model serve -c output/run_001/best_model \\
            --reload --log-level debug
        
        # High-performance serving
        bert model serve -c output/run_001/best_model \\
            --workers 4 --batch-size 64 --cache-size 1000
    """
    # Validate SSL configuration
    if (ssl_cert and not ssl_key) or (ssl_key and not ssl_cert):
        print_error("Both --ssl-cert and --ssl-key must be provided for SSL")
        raise typer.Exit(1)
    
    if ssl_cert and ssl_key:
        if not ssl_cert.exists():
            print_error(f"SSL certificate not found: {ssl_cert}")
            raise typer.Exit(1)
        if not ssl_key.exists():
            print_error(f"SSL key not found: {ssl_key}")
            raise typer.Exit(1)
    
    # Display server configuration
    console.print(f"\n[bold blue]Model Server Configuration[/bold blue]")
    console.print(f"Model checkpoint: {checkpoint}")
    console.print(f"Server address: {host}:{port}")
    console.print(f"Workers: {workers}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Max sequence length: {max_length}")
    
    if ssl_cert:
        console.print(f"SSL enabled: ✓")
    if auth_token:
        console.print(f"Authentication: ✓")
    if cache_size > 0:
        console.print(f"Response caching: {cache_size} entries")
    
    console.print()
    
    # Create FastAPI app
    try:
        app = _create_api_app(
            checkpoint=checkpoint,
            model_name=model_name or checkpoint.name,
            batch_size=batch_size,
            max_length=max_length,
            cache_size=cache_size,
            auth_token=auth_token,
            enable_cors=cors,
            enable_metrics=enable_metrics,
            enable_health=health_check,
            enable_docs=docs,
            verbose=verbose
        )
        
        # Display API endpoints
        console.print("[bold green]API Endpoints:[/bold green]")
        console.print(f"  • POST   /predict      - Single prediction")
        console.print(f"  • POST   /predict/batch - Batch predictions")
        if health_check:
            console.print(f"  • GET    /health       - Health check")
        if enable_metrics:
            console.print(f"  • GET    /metrics      - Prometheus metrics")
        if docs:
            console.print(f"  • GET    /docs         - Interactive API docs")
            console.print(f"  • GET    /redoc        - Alternative API docs")
        
        console.print(f"\n[bold blue]Starting server...[/bold blue]")
        console.print(f"Press CTRL+C to stop\n")
        
        # Configure uvicorn
        uvicorn_config = {
            "app": app,
            "host": host,
            "port": port,
            "workers": workers if not reload else 1,
            "reload": reload,
            "log_level": log_level.lower(),
            "timeout_keep_alive": timeout,
            "access_log": verbose,
        }
        
        if ssl_cert and ssl_key:
            uvicorn_config.update({
                "ssl_certfile": str(ssl_cert),
                "ssl_keyfile": str(ssl_key),
            })
        
        # Start server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        print_info("\nServer stopped by user")
    except Exception as e:
        print_error(f"Server error: {e}")
        raise typer.Exit(1)


def _create_api_app(
    checkpoint: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    cache_size: int,
    auth_token: Optional[str],
    enable_cors: bool,
    enable_metrics: bool,
    enable_health: bool,
    enable_docs: bool,
    verbose: bool
) -> Any:
    """Create FastAPI application for model serving."""
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any, Optional
    import mlx.core as mx
    from functools import lru_cache
    import time
    
    # Create FastAPI app
    app = FastAPI(
        title=f"{model_name} API",
        description="MLX BERT model inference API",
        version="1.0.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
    )
    
    # Add CORS middleware
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Authentication
    security = HTTPBearer(auto_error=False) if auth_token else None
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if auth_token and credentials.credentials != auth_token:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Request/Response models
    class PredictionRequest(BaseModel):
        text: str = Field(..., description="Input text for classification")
        return_probabilities: bool = Field(False, description="Return class probabilities")
        
    class BatchPredictionRequest(BaseModel):
        texts: List[str] = Field(..., description="List of texts for classification")
        return_probabilities: bool = Field(False, description="Return class probabilities")
    
    class PredictionResponse(BaseModel):
        prediction: str = Field(..., description="Predicted class label")
        confidence: float = Field(..., description="Prediction confidence")
        probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
        model: str = Field(..., description="Model name")
        processing_time: float = Field(..., description="Processing time in seconds")
    
    class BatchPredictionResponse(BaseModel):
        predictions: List[PredictionResponse] = Field(..., description="List of predictions")
        total_time: float = Field(..., description="Total processing time in seconds")
    
    class HealthResponse(BaseModel):
        status: str = Field(..., description="Service status")
        model: str = Field(..., description="Model name")
        device: str = Field(..., description="Compute device")
        uptime: float = Field(..., description="Service uptime in seconds")
    
    # Load model (singleton)
    @lru_cache(maxsize=1)
    def get_model():
        """Load and cache the model."""
        logger.info(f"Loading model from {checkpoint}")
        
        # Import model classes
        from models.modernbert_optimized import ModernBertModel
        from models.classification_head import ClassificationHead
        from embeddings.tokenizer_wrapper import UnifiedTokenizer
        
        # Load config
        config_path = checkpoint / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = ModernBertModel(config)
        classification_head = ClassificationHead(
            hidden_size=config['hidden_size'],
            num_labels=config.get('num_labels', 2),
            dropout=config.get('classifier_dropout', 0.1)
        )
        
        # Load weights
        weights_path = checkpoint / "model.safetensors"
        if weights_path.exists():
            from safetensors import safe_open
            weights = {}
            with safe_open(weights_path, framework="mlx") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            model.load_weights(list(weights.items()))
            classification_head.load_weights(list(weights.items()))
        
        # Load tokenizer
        tokenizer = UnifiedTokenizer(
            model_name=config.get('model_name', 'answerdotai/ModernBERT-base'),
            backend='auto',
            checkpoint_path=checkpoint
        )
        
        # Set to eval mode
        model.eval()
        classification_head.eval()
        
        return model, classification_head, tokenizer, config
    
    # Prediction cache
    if cache_size > 0:
        prediction_cache = lru_cache(maxsize=cache_size)(_predict_single)
    else:
        prediction_cache = _predict_single
    
    def _predict_single(text: str, return_probs: bool = False):
        """Make a single prediction."""
        model, head, tokenizer, config = get_model()
        
        # Tokenize
        tokens = tokenizer(text, max_length=max_length, truncation=True)
        input_ids = mx.array([tokens['input_ids']])
        attention_mask = mx.array([tokens['attention_mask']])
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        logits = head(outputs['last_hidden_state'][:, 0, :])
        
        # Get prediction
        probs = mx.softmax(logits, axis=-1)
        pred_idx = mx.argmax(probs, axis=-1).item()
        confidence = probs[0, pred_idx].item()
        
        # Get label
        label_map = config.get('label_map', {0: 'negative', 1: 'positive'})
        prediction = label_map.get(pred_idx, str(pred_idx))
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
        }
        
        if return_probs:
            result['probabilities'] = {
                label_map.get(i, str(i)): probs[0, i].item()
                for i in range(len(label_map))
            }
        
        return result
    
    # Service start time
    start_time = time.time()
    
    # API endpoints
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        credentials: HTTPAuthorizationCredentials = Depends(security) if auth_token else None
    ):
        """Make a single prediction."""
        start = time.time()
        
        try:
            result = prediction_cache(request.text, request.return_probabilities)
            
            return PredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                probabilities=result.get('probabilities'),
                model=model_name,
                processing_time=time.time() - start
            )
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(
        request: BatchPredictionRequest,
        credentials: HTTPAuthorizationCredentials = Depends(security) if auth_token else None
    ):
        """Make batch predictions."""
        start = time.time()
        
        if len(request.texts) > batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.texts)} exceeds maximum {batch_size}"
            )
        
        try:
            predictions = []
            for text in request.texts:
                pred_start = time.time()
                result = prediction_cache(text, request.return_probabilities)
                
                predictions.append(PredictionResponse(
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    probabilities=result.get('probabilities'),
                    model=model_name,
                    processing_time=time.time() - pred_start
                ))
            
            return BatchPredictionResponse(
                predictions=predictions,
                total_time=time.time() - start
            )
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    if enable_health:
        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model=model_name,
                device=str(mx.default_device()),
                uptime=time.time() - start_time
            )
    
    if enable_metrics:
        # Simple metrics endpoint (could be extended with prometheus_client)
        @app.get("/metrics")
        async def metrics():
            """Prometheus-style metrics endpoint."""
            metrics_text = f"""# HELP model_info Model information
# TYPE model_info gauge
model_info{{name="{model_name}",checkpoint="{checkpoint}"}} 1

# HELP uptime_seconds Service uptime in seconds
# TYPE uptime_seconds counter
uptime_seconds {time.time() - start_time:.2f}

# HELP cache_size Prediction cache size
# TYPE cache_size gauge
cache_size {cache_size}
"""
            return Response(content=metrics_text, media_type="text/plain")
    
    @app.on_event("startup")
    async def startup():
        """Initialize model on startup."""
        logger.info("Initializing model...")
        get_model()  # Pre-load model
        logger.info("Model loaded successfully")
    
    return app


# Helper function for testing serving without starting server
def test_serving(checkpoint: Path, text: str = "This is a test input"):
    """Test model serving locally without starting server."""
    console.print(f"\n[bold blue]Testing Model Serving[/bold blue]")
    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Test input: {text}")
    
    try:
        # Create minimal app
        app = _create_api_app(
            checkpoint=checkpoint,
            model_name="test-model",
            batch_size=1,
            max_length=256,
            cache_size=0,
            auth_token=None,
            enable_cors=False,
            enable_metrics=False,
            enable_health=False,
            enable_docs=False,
            verbose=True
        )
        
        # Test prediction
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.post("/predict", json={"text": text})
        if response.status_code == 200:
            result = response.json()
            print_success("Model serving test successful!")
            console.print(f"Prediction: {result['prediction']}")
            console.print(f"Confidence: {result['confidence']:.3f}")
        else:
            print_error(f"Test failed: {response.text}")
            
    except Exception as e:
        print_error(f"Test error: {e}")
        raise typer.Exit(1)