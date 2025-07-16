"""Test basic MLX functionality with our model."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Test 1: Basic array operations
print("Test 1: Basic MLX array operations")
a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])
c = a + b
print(f"Array addition: {a} + {b} = {c}")

# Test 2: Simple model
print("\nTest 2: Simple MLX model")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def __call__(self, x):
        return self.linear(x)

model = SimpleModel()
x = mx.random.normal((4, 10))  # batch_size=4, input_dim=10
y = model(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")

# Test 3: Loss and gradients
print("\nTest 3: Loss and gradient computation")
def loss_fn(model, x, y_true):
    y_pred = model(x)
    # Simple MSE loss
    return mx.mean((y_pred - y_true) ** 2)

y_true = mx.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=mx.float32)

# Compute loss
loss_val = loss_fn(model, x, y_true)
print(f"Loss: {loss_val}")

# Compute gradients - MLX way
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss_val, grads = loss_and_grad_fn(model, x, y_true)
print(f"Loss with gradients: {loss_val}")
print(f"Gradient keys: {list(grads.keys())}")

# Test 4: Optimizer update
print("\nTest 4: Optimizer update")
optimizer = optim.SGD(learning_rate=0.01)
optimizer.update(model, grads)
print("Parameters updated!")

# Test 5: Our model structure
print("\nTest 5: Testing our TitanicClassifier")
from models.modernbert_mlx import ModernBertConfig, ModernBertMLX
from models.classification_head import TitanicClassifier

# Create a small config for testing
config = ModernBertConfig(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=128
)

bert = ModernBertMLX(config)
classifier = TitanicClassifier(bert)

# Test forward pass
input_ids = mx.array(np.random.randint(0, 100, (2, 10)))  # batch_size=2, seq_len=10
attention_mask = mx.ones_like(input_ids)
labels = mx.array([0, 1])

print(f"Input shape: {input_ids.shape}")

# Forward pass
outputs = classifier(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(f"Loss: {outputs['loss']}")
print(f"Logits shape: {outputs['logits'].shape}")

print("\nAll tests passed!")