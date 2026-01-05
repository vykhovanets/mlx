# MLX Learning Strategy: From Beginner to Apple Contributor

A strategic, phased approach to mastering MLX, contributing to Apple, and building agentic AI systems.

---

## Your Goals Mapped to Action

| Goal | Path |
|------|------|
| Become MLX contributor | Phase 2-3 contributions |
| Understand AI/ML deeply | Phase 1 fundamentals + implementation |
| Train CV models on Mac | Phase 1-2 projects |
| Agentic workflows + MLX | Phase 3 capstone |

---

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Install and Run MLX

```bash
# Create isolated environment
python3 -m venv mlx-env
source mlx-env/bin/activate
pip install mlx mlx-lm
```

### 1.2 Core Concepts to Master

**Start with the linear regression example** (`examples/python/linear_regression.py`):

```python
import mlx.core as mx

# Key insight: MLX is lazy - nothing runs until mx.eval()
x = mx.random.normal((100, 10))
y = mx.sum(x)  # Not computed yet!
mx.eval(y)     # Now it runs

# Automatic differentiation - THE core ML primitive
def loss_fn(w):
    return mx.mean(mx.square(predictions - targets))

grad_fn = mx.grad(loss_fn)  # Creates gradient function
gradients = grad_fn(weights)
```

**Critical concepts:**
1. **Lazy evaluation** - Computation graphs build up, execute on `mx.eval()`
2. **Unified memory** - No CPU/GPU copies needed on Apple Silicon
3. **Function transforms** - `mx.grad()`, `mx.vmap()`, `mx.jit()`

### 1.3 First Project: MNIST from Scratch

Build this yourself to understand the fundamentals:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = self.pool(nn.relu(self.conv1(x)))
        x = self.pool(nn.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)

# Training loop pattern
model = SimpleCNN()
optimizer = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

loss_and_grad = nn.value_and_grad(model, loss_fn)

for batch in dataloader:
    loss, grads = loss_and_grad(model, batch['image'], batch['label'])
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

**Data loading challenge:** MLX lacks built-in DataLoader - this is your first contribution opportunity!

---

## Phase 2: Computer Vision Focus (Weeks 3-4)

### 2.1 Use mlx-image for Real Training

```bash
pip install mlx-image
```

The mlx-image library provides:
- PyTorch-like DataLoader and Dataset
- Pre-trained models (ResNet, ViT, EfficientNet, MobileNet)
- Training utilities

**Project: Train CIFAR-10 classifier**

```python
from mlxim.model import create_model
from mlxim.data import LabelFolderDataset, DataLoader

# Load pretrained ResNet18, fine-tune on your data
model = create_model("resnet18", pretrained=True, num_classes=10)

dataset = LabelFolderDataset(
    root="path/to/cifar10",
    class_map={"airplane": 0, "automobile": 1, ...}
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2.2 Contribution Opportunity: Image Transforms

**Gap identified:** mlx-image lacks comprehensive transforms like torchvision.

Create `mlx/transforms.py`:

```python
# This would be a great first PR!
import mlx.core as mx

def normalize(x, mean, std):
    """Normalize image tensor."""
    mean = mx.array(mean).reshape(1, 1, 1, -1)
    std = mx.array(std).reshape(1, 1, 1, -1)
    return (x - mean) / std

def random_horizontal_flip(x, p=0.5):
    """Randomly flip image horizontally."""
    if mx.random.uniform() < p:
        return mx.flip(x, axis=-2)  # Flip width axis
    return x

def random_crop(x, size, padding=0):
    """Random crop with optional padding."""
    # Implementation...
```

### 2.3 Models to Implement (Contribution Ideas)

| Model | Complexity | Impact | Notes |
|-------|------------|--------|-------|
| LeNet-5 | Low | Educational | Classic, ~60 lines |
| ResNet-18 | Medium | High | Most requested |
| MobileNetV3 | Medium | High | Efficient inference |
| EfficientNet-B0 | Medium-High | High | State-of-art efficiency |
| ViT-Tiny | Medium | High | Transformers for vision |

---

## Phase 3: Agentic Workflows + MLX (Weeks 5-8)

### 3.1 The Vision: Local AI Agent with MLX Backend

Combine your interests:
- **MLX** for fast local inference
- **Small LLMs** (Qwen 0.5B, Llama 3.2 1B/3B) for reasoning
- **Tool use** for agentic behavior
- **Vision models** via mlx-vlm for multimodal capabilities

### 3.2 Architecture

```
┌─────────────────────────────────────────────────┐
│                  Your Agent                      │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────┐    │
│  │   mlx-lm    │    │      mlx-vlm        │    │
│  │ (reasoning) │    │ (vision understanding)│   │
│  └──────┬──────┘    └──────────┬──────────┘    │
│         │                      │                │
│         └──────────┬───────────┘                │
│                    │                            │
│         ┌──────────▼──────────┐                │
│         │    Tool Router      │                │
│         └──────────┬──────────┘                │
│                    │                            │
│    ┌───────────────┼───────────────┐           │
│    │               │               │           │
│    ▼               ▼               ▼           │
│ ┌──────┐      ┌──────┐      ┌──────────┐      │
│ │ Bash │      │ Read │      │ CV Model │      │
│ │ Tool │      │ Tool │      │ (mlx-image)│     │
│ └──────┘      └──────┘      └──────────┘      │
└─────────────────────────────────────────────────┘
```

### 3.3 Starter Implementation

```python
# agent.py - Minimal agentic loop with MLX
from mlx_lm import load, generate

class MLXAgent:
    def __init__(self, model_name="mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        self.model, self.tokenizer = load(model_name)
        self.tools = {
            "analyze_image": self.analyze_image,
            "run_code": self.run_code,
            "search_files": self.search_files,
        }

    def run(self, user_query: str, max_turns: int = 10):
        messages = [{"role": "user", "content": user_query}]

        for _ in range(max_turns):
            response = self.generate_response(messages)

            if tool_call := self.extract_tool_call(response):
                result = self.execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result: {result}"})
            else:
                return response  # Final answer

        return "Max turns reached"

    def generate_response(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=512)
```

### 3.4 Learn from learn-claude-code Repo

The [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) repo teaches:
- **v0-v4 progressive agent implementations** (~1100 lines total)
- Core insight: "Model is 80%, code is 20%"
- Planning with todo management
- Subagent spawning
- Skills system

Study their patterns, then implement with MLX backend instead of API calls.

---

## Phase 4: Contributing to MLX (Ongoing)

### 4.1 High-Impact Contribution Areas

| Area | Files to Study | Impact |
|------|----------------|--------|
| **DataLoader** | Create `python/mlx/data/` | Critical gap |
| **Image transforms** | `python/mlx/nn/` | Vision users need this |
| **Loss functions** | `python/mlx/nn/losses.py` | Add focal, dice, contrastive |
| **Metrics** | Create `python/mlx/metrics/` | Training utilities |
| **Vision models** | Create `python/mlx/vision/` | Model zoo |

### 4.2 Your First PR Checklist

1. **Fork the repo** and create feature branch
2. **Study existing patterns** in `python/mlx/nn/layers/`
3. **Write tests** (see `python/tests/`)
4. **Run formatters**: `pre-commit run --all-files`
5. **Benchmark** if performance-critical
6. **Submit PR** with clear description

### 4.3 Suggested First Contributions (Ordered by Difficulty)

1. **Add FocalLoss** to `nn/losses.py` (~20 lines)
2. **Add DiceLoss** for segmentation (~30 lines)
3. **Create basic transforms** (normalize, random_flip)
4. **Implement LeNet-5** as example model
5. **Write MNIST training example**
6. **Build DataLoader utility class**

---

## Concrete Weekly Plan

### Week 1: Fundamentals
- [ ] Install MLX, run examples
- [ ] Implement linear regression from scratch (no copying!)
- [ ] Read: lazy_evaluation.rst, function_transforms.rst

### Week 2: First Neural Network
- [ ] Build MNIST classifier with `nn.Module`
- [ ] Understand `nn.value_and_grad()` pattern
- [ ] Implement custom loss function

### Week 3: Computer Vision Deep Dive
- [ ] Install and explore mlx-image
- [ ] Train CIFAR-10 with ResNet18
- [ ] Study Conv2d implementation in mlx source

### Week 4: First Contribution
- [ ] Fork MLX repo
- [ ] Implement one missing loss function (FocalLoss)
- [ ] Write tests, submit PR

### Week 5-6: Agentic Foundations
- [ ] Install mlx-lm, run Qwen 0.5B locally
- [ ] Study learn-claude-code repo patterns
- [ ] Build minimal agent loop with tool use

### Week 7-8: Capstone Project
- [ ] Build vision-enabled agent with mlx-vlm
- [ ] Integrate custom CV model for specific task
- [ ] Document and share your work

---

## Resources

### Official
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [WWDC 2025: Get Started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)

### Ecosystem
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference/fine-tuning
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-image](https://github.com/riccardomusmeci/mlx-image) - Computer vision
- [HuggingFace mlx-community](https://huggingface.co/mlx-community) - Model weights

### Learning Agents
- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) - Agent patterns

---

## Why This Strategy Works

1. **Start small, compound learning** - Each phase builds on the last
2. **Learn by implementing** - Understanding comes from building, not reading
3. **Contribute while learning** - PRs force deep understanding
4. **Practical capstone** - Combines all skills into something useful
5. **Apple visibility** - Quality contributions get noticed

The key insight: **Don't try to understand everything first.** Pick a tiny piece, implement it, then expand. The mlx codebase is well-organized - you can understand `nn.losses` without knowing Metal kernels.

Good luck! Start with Week 1 today.
