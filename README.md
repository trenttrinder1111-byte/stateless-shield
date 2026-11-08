# Stateless Shield 🛡️
*High-Performance, Sub-Millisecond Vector Memory Engine*

Stateless Shield is a C++ backed, SIMD-accelerated memory retrieval engine designed for real-time AI and embedded systems. It bypasses Python interpreter overhead to deliver sub-linear $O(\log n)$ performance.

## Key Performance Benchmarks
* **Query Latency:** $\approx 0.25$ ms at 10,000 nodes; $\approx 0.59$ ms at 50,000 nodes.
* **Memory Footprint:** $\sim 5$ MB RAM usage via native `f16` scalar quantization.
* **Dependency Footprint:** Lightweight, header-only C++ core with zero heavy external infrastructure.

## Quickstart

```python
import numpy as np
from shield import StatelessShield

# Initialize engine with 128-dimensional vectors
shield = StatelessShield(vector_dim=128)

# Add a node
vector = np.random.randn(128)
shield.add(node_id=1, vector=vector, payload="Context chunk 1")

# Search instantly
results = shield.search(query_vector=vector, top_k=1)
print(results)
