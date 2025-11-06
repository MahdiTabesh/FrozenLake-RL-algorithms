# FrozenLake-RL-algorithms

This repository contains modular implementations of key reinforcement learning algorithms applied to OpenAI Gym's `FrozenLake-v1` environment. Developed as part of the Reinforcement Learning course (ELEN E6885) at Columbia University.

---

## 📁 Contents

### `/RLalgs/`
- `pi.py` — Policy Iteration
- `vi.py` — Value Iteration
- `ql.py` — Q-Learning
- `sarsa.py` — SARSA
- `utils.py` — Shared helpers and evaluation logic
- `__init__.py` — Makes `RLalgs` a package

### `FrozenLake.ipynb`
Interactive Colab-compatible notebook:
- Imports and evaluates all RL algorithms
- Includes policy rendering, reward visualization, and convergence tracking
- Used to generate the PDF report for submission

---

## Environment
- Python 3.12+
- `gymnasium` or `gym`
- `numpy`
- Optional: `matplotlib` for visualization

---

## 👤 Author
**Mahdi Tabesh**  
M.S. Student, Columbia University  
Course: ELEN E6885 - Reinforcement Learning

