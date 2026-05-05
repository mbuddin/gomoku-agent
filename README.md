#Adversarial Decision-Making System for Gomoku

A Python-based Gomoku (Five-in-a-Row) game-playing agent featuring a depth-limited
Minimax search with Alpha-Beta pruning, Transposition Table, Killer-Move, and
History heuristics, backed by a pattern-based evaluation function.
Playable on a 15×15 board.

## Project Structure

```
Gomoku-Agent/
├── gomoku/                    # Core package
│   ├── __init__.py
│   ├── board.py               # Board representation, Zobrist hashing, win/draw detection
│   ├── heuristic.py           # Pattern-based heuristic evaluation (open/rush fours, threes, …)
│   ├── move_gen.py            # Candidate move generation (neighbour radius)
│   ├── search.py              # Minimax + Alpha-Beta + TT + Killer + History
│   └── game.py                # Game state manager (move, undo, win/draw detection)
├── tests/                     # Unit tests
│   ├── test_board.py
│   ├── test_heuristic.py
│   ├── test_move_gen.py
│   └── test_search.py
├── main.py                    # pygame GUI entry point
├── performance_metrics.py     # Experiment 1: decision time, win-rate, game length
├── benchmark.py               # Experiment 2: AI-vs-AI depth matrix benchmark
├── benchmark_algo.py          # Experiment 3: Minimax vs Alpha-Beta comparison
├── analyze.py                 # Analyze & print benchmark.py results
├── analyze_algo.py            # Visualize benchmark_algo.py results (matplotlib)
├── results/                   # Saved benchmark outputs (CSV + TXT)
├── requirements.txt           # Runtime + development dependencies
└── pyproject.toml             # Project metadata and tooling configuration
```

## Setup

**Prerequisites:** Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone 
cd Gomoku-Agent

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

On launch a **mode-selection menu** appears:

| Mode | Description |
|---|---|
| **Player vs AI** | You play as Black; AI (White, depth-3 Alpha-Beta) responds automatically |
| **AI vs AI** | Launches `benchmark.py` as a subprocess |

**Controls (Player vs AI):**

| Action | Input |
|---|---|
| Place a stone | Left-click a grid intersection |
| Undo last full turn | Click **Undo** · or press `U` / `Backspace` |
| Restart | Click **Restart** · or press `R` / `Escape` |

## Running Tests

```bash
pytest tests/ -v
```

## Experiment Scripts

### Experiment 1 — Performance Metrics

Measures AI decision time, win rate vs a random player, and average game length.

```bash
python performance_metrics.py
```

### Experiment 2 — AI vs AI Benchmark

Runs AI-vs-AI games across all depth combinations and reports win-rate matrix and
search efficiency (nodes, pruning rate, EBF). Results saved to `results/benchmark/`.

```bash
python benchmark.py --depths 1 2 3 --games 3
python analyze.py --csv results/benchmark/<latest>.csv
```

### Experiment 3 — Minimax vs Alpha-Beta

Compares plain Minimax against Alpha-Beta at equal depth, quantifying node reduction,
pruning rate, and speedup. Results saved to `results/benchmark_algo/`.

```bash
python benchmark_algo.py --depths 1 2 3 --games 3
python analyze_algo.py results/benchmark_algo/<latest>.csv
```
