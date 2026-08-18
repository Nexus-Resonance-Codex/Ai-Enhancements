# Executive Agent (`ExecutiveAgent`)

## 1. Overview
High-level task routing coordinator matching problem descriptions to optimal solver targets across the NRC manifold.

## 2. PyTorch Implementation
```python
from nrc_ai import ExecutiveAgent

agent = ExecutiveAgent()
target = agent.route_task('Fold protein with structural constraints')
assert target is not None
```
