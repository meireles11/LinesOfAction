# Lines of Action

A Python implementation of the board game *Lines of Action*, created as part of a university project.  
Includes two game modes: Player vs Player and Player vs AI.

## About the Game

*Lines of Action* (LOA) is a two-player abstract strategy board game played on an 8×8 board.  
The objective is to connect all of your pieces into a single group (orthogonally or diagonally connected).

## Movement Rules

- A piece moves in a straight line (horizontal, vertical, or diagonal).
- The distance moved equals the total number of pieces (both yours and your opponent’s) on that line.
- You can jump over your own pieces, but cannot jump over opponent’s pieces.
- You can land on an opponent’s piece, in which case you capture it.

For full rules, see: [Wikipedia – Lines of Action](https://en.wikipedia.org/wiki/Lines_of_Action)



## Features

- Turn-based gameplay on an 8x8 board
- Valid move highlighting
- AI opponent using Minimax algorithm (with adjustable difficulty)
- Hint system: when the player clicks on the piece that the AI considers best to move, the suggested best move is highlighted in blue.
- Two modes:
  - `player_vs_AI.py`
  - `player_vs_player.py`

## Requirements

- Python 3.x
- `pygame` library

Install dependencies:
```bash
pip install pygame
```

## How to Play

Run one of the following files depending on the mode you want:

```bash
python player_vs_player.py  # Two players on same machine
python player_vs_ai.py      # Play against the AI
```

