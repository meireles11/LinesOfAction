import pygame
import math, random
from copy import deepcopy
from itertools import combinations
import math

# Center of Mass evaluation function
def center_zone_spread(board, player):
  n = len(board)
  center_of_mass = [(n//2-1,n//2-1),(n//2,n//2-1),(n//2-1,n//2),(n//2,n//2)]
  result = 0
  for y in range(n):
    for x in range(n):
      if (board[y][x] == player):
        result += min(math.sqrt((x-cx)**2 + (y-cy)**2) for cx,cy in center_of_mass)
  return int(result)
  

# Control over opponent evaluation function
def control_over_opponent(board, player):
  if player == 'W':
    opponent = 'B'
  else:
    opponent = 'W'
  number = len(valid_moves(board,opponent))
  n = len(board)
  count = 0
  for y in range(n):
     for x in range(n):
        if (board[y][x] == opponent):
          count += 1
  return number/count



# Returns all legal moves for a player
def valid_moves(board, player):
  n = len(board)
  directions = [(1,0),(0,1),(1,1),(1,-1)]
  moves = []
  
  for y in range(n):
    for x in range(n):
      if (board[y][x] == player):
        for dx,dy in directions:
          count = 0
          
          # Count pieces in the positive direction
          
          tx,ty = x,y
          while 0<=tx<n and 0<=ty<n:
            if (board[ty][tx] != '.'):
              count += 1
            tx += dx
            ty += dy
          
          # Count pieces in the negative direction
          
          tx,ty = x-dx,y-dy
          while 0<=tx<n and 0<=ty<n:
            if (board[ty][tx] != '.'):
              count += 1
            tx -= dx
            ty -= dy
            
          #Check both directions
          
          for k in range(2):
            tx,ty = x,y
            blocked = False
            for i in range(count):
              tx += dx*(-1)**k
              ty += dy*(-1)**k
              
              # If it moves out of bounds, the move is blocked
              if (tx<0 or tx>n-1 or ty<0 or ty>n-1):
                blocked = True
                break

              # If an opponent's piece is encountered before the last step, the move is blocked
              if (board[ty][tx] != '.' and board[ty][tx] != player and i != count-1):
                blocked = True
                break

              # If a player's own piece is at the final destination, the move is blocked
              if (board[ty][tx] == player and i == count-1):
                blocked = True
                break
            if (blocked == False):
              moves += [(x,y,tx,ty)]
  
  return moves

            
            

# Auxiliary function of distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Group Compactness evaluation function
def medium_distance_evaluation(board, player):
    pieces = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == player:
                pieces.append((i,j))
    
    if len(pieces) < 2:
        return 0
    pairs = list(combinations(pieces, 2))

    distance_sum = 0
    for p1, p2 in pairs:
        distance_sum += (distance(p1, p2)) 

    medium_distance = distance_sum / len(pairs)

    return medium_distance 


# Combination of evaluation functions
def combined_evaluation(board, player):
    w1, w2, w3 = -1.0, 1, -1  
    
    center_value = center_zone_spread(board, player)
    opponent_control = control_over_opponent(board, player)
    compactness = medium_distance_evaluation(board, player)
    
    return w1 * center_value + w2 * opponent_control + w3 * compactness

def flood_fill(state, r, c, visited, player):
    board = state.board
    size = state.size
    
    if (r, c) in visited or board[r][c] != player:
        return 0
    
    visited.add((r, c))
    count = 1  
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] #Possible moviments in every direction
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size: #Checks the board's limits
            count += flood_fill(state, nr, nc, visited, player)
    
    return count

#Returns True if a player has a winning position
def is_winner(state,player):
    board = state.board
    size = state.size

    positions = []
    for r in range(size):
        for c in range(size):
            if board[r][c] == player:
                positions.append((r,c))
    
    #If one player has only one piece wins automatically
    if len(positions) == 1:
        return True  

    visited = set()
    #Checks how many pieces are connected
    connected_pieces = flood_fill(state, positions[0][0], positions[0][1], visited, player)

    #If the total number of pieces is the same as the  number of connected pieces returns True
    if connected_pieces == len(positions):
        return True
    else:
        return False

#If both players reach a winning psoition at the same time the moving player wins
def check_victory(state):
    current_player = state.player

    player_B_wins = is_winner(state, 'B')
    player_W_wins = is_winner(state, 'W')

    if player_B_wins and player_W_wins:
        return current_player 
    elif player_B_wins:
        return 'B'
    elif player_W_wins:
        return 'W'
    
    return None #the game continues

# Defining constants
TILE_SIZE = 60
BOARD_SIZE = 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)

# Pieces' colors
PIECES = {
    'B': BLACK,  # Black
    'W': WHITE,  # White
}

class LineOfAction:
    def __init__(self):
        self.size = BOARD_SIZE  
        self.board = self.initialize_board()  
        self.player = 'B'

    def initialize_board(self):
        board = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for i in range(1, self.size - 1):
            board[0][i] = 'B'  
            board[self.size - 1][i] = 'B'  
            board[i][0] = 'W'  
            board[i][self.size - 1] = 'W'  
        return board

    def draw_board(self, screen):
        # Draws the board and its pieces
        for y in range(self.size):
            for x in range(self.size):
                pygame.draw.rect(screen, BROWN, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)) #Fills the board with brown 
                pygame.draw.rect(screen, BLACK, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3) #Draws the boards with black
                piece = self.board[y][x]
                if piece != '.':
                    pygame.draw.circle(screen, PIECES[piece], (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 3) #Dras the pieces

    def display_board(self, screen):
        self.draw_board(screen)
        pygame.display.flip() #Atualizes the board
    
    def apply_move(self, move):
        xi, yi, xf, yf = move
        self.board[yf][xf] = self.board[yi][xi]
        self.board[yi][xi] = '.'
        self.player = 'B' if self.player == 'W' else 'W'  # Switch Player
    
    def reset(self):
        self.board = self.initialize_board()
        self.player = 'B'



def apply_move_copy(state, move):
    # Applies a move to the state and returns a new state without modifying the original and without switching players.
    new_state = deepcopy(state)  # Create a copy of the state
    xi, yi, xf, yf = move
    new_state.board[yf][xf] = new_state.board[yi][xi]
    new_state.board[yi][xi] = '.'
    return new_state

def minimax(state, depth, maximizing_player, alpha, beta):
    # Implementation of the Minimax algorithm with alpha-beta pruning.
    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves(state.board, state.player):
            new_state = apply_move_copy(state, move)
            winner = check_victory(state)
            if winner is not None or depth == 0:
              return combined_evaluation(state.board, state.player)
            new_state.player = 'B' if new_state.player == 'W' else 'W'
            evaluation = minimax(new_state, depth - 1, False, alpha, beta)
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in valid_moves(state.board, state.player):
            new_state = apply_move_copy(state, move)
            winner = check_victory(state)
            if winner is not None or depth == 0:
              return combined_evaluation(state.board, state.player)
            new_state.player = 'B' if new_state.player == 'W' else 'W'
            evaluation = minimax(new_state, depth - 1, True, alpha, beta)
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval

def best_move(state, depth):
    # Returns the best move to the current player with Minimax
    best_move = None
    alpha, beta = -math.inf, math.inf
    maximizing = state.player == 'B'
    best_score = -math.inf if maximizing else math.inf

    for move in valid_moves(state.board, state.player):
        new_state = apply_move_copy(state, move)
        new_state.player = 'B' if new_state.player == 'W' else 'W'
        score = minimax(new_state, depth - 1, not maximizing, alpha, beta)

        if (maximizing and score > best_score) or (not maximizing and score < best_score):
            best_score = score
            best_move = move

        if maximizing:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)

        if beta <= alpha:
            break
    
    return best_move



def mcts_search(state, simulations=1000):
    #Simplified implementation of Monte Carlo Tree Search (MCTS).
    # Performs random simulations and selects the move that led to the most victories.

    move_scores = {move: 0 for move in valid_moves(state.board, state.player)}

    for _ in range(simulations):
        move = random.choice(list(move_scores.keys()))
        new_state = apply_move_copy(state, move)
        new_state.player = 'B' if new_state.player == 'W' else 'W'
        result = simulate_game(new_state)
        move_scores[move] += result

    # Chose the move with the highest score in the simulations
    best_move = max(move_scores, key=move_scores.get)
    return best_move

def simulate_game(state):
    """
    Simulates a random game from the current state and returns:
        1 if the current player wins
        -1 if the opponent wins
        0 if it's a draw
    """
    player = state.player
    while True:
        moves = valid_moves(state.board, player)
        if not moves:
            return 0  # Draw

        move = random.choice(moves)
        state = apply_move_copy(state, move)

        # Checks if there is a winner after the move 
        winner = check_victory(state)
        if winner is not None:
            return 1 if winner == player else -1  # The player that did the last move wins

        player = 'B' if player == 'W' else 'W'  # Change player

def strategy(state, depth=3):
    # Chooses whether to use Minimax or Monte Carlo to determine the best move.
    # Uses Minimax for early-game states (few moves made) and MCTS for more complex states.

    num_moves = len(valid_moves(state.board, state.player))

    if num_moves <= 20: 
        return best_move(state, depth)  # Minimax
    else:
        return mcts_search(state, simulations=1000)  # MCTS



def main():
    pygame.init()
    WIDTH = TILE_SIZE * BOARD_SIZE  # Define before creating the screen
    HEIGHT = TILE_SIZE * BOARD_SIZE
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Lines of Action - Human against Human")

    game = LineOfAction()
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    button_font = pygame.font.SysFont(None, 36)  # Separate font for the button

    # Variables for interface control
    selected_piece = None
    valid_targets = []
    current_player = game.player
    winner = None

    running = True
    while running:
        screen.fill((0, 0, 0))
        game.display_board(screen)

        # Draw highlight for the selected piece
        if selected_piece:
            x, y = selected_piece
            pygame.draw.rect(screen, (0, 255, 0), 
                           (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

        # Draw highlight for valid moves
        for target in valid_targets:
            tx, ty = target
            pygame.draw.rect(screen, (255, 255, 0), 
                           (tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

        winner = check_victory(game)
        
        if winner:
            message = "White wins!" if winner == "W" else "Black wins!"
            text_surface = font.render(message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
            screen.blit(text_surface, text_rect)

            # Reset button
            GREEN = (0, 255, 0)
            WHITE = (255, 255, 255)
            button_text = button_font.render("Reset", True, WHITE)
            text_rect = button_text.get_rect()
            padding_x = 20
            padding_y = 10
            button_rect = pygame.Rect(
                WIDTH // 2 - (text_rect.width + padding_x) // 2,
                HEIGHT // 2 + 20,
                text_rect.width + padding_x,
                text_rect.height + padding_y
            )
            pygame.draw.rect(screen, GREEN, button_rect)
            screen.blit(
                button_text,
                (button_rect.x + padding_x // 2,
                 button_rect.y + padding_y // 2)
            )
            
            pygame.display.flip()

            # Wait for button click
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect.collidepoint(event.pos):
                            # Reset game
                            game.reset()
                            selected_piece = None
                            valid_targets = []
                            winner = None
                            current_player = game.player
                            waiting = False

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not winner and event.type == pygame.MOUSEBUTTONDOWN:
                # Get click position
                mouse_x, mouse_y = pygame.mouse.get_pos()
                col = mouse_x // TILE_SIZE
                row = mouse_y // TILE_SIZE

                # If no piece is selected, try to select one belonging to the current player
                if selected_piece is None:
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                        if game.board[row][col] == current_player:
                            selected_piece = (col, row)
                            # Find all valid moves for this piece
                            all_moves = valid_moves(game.board, current_player)
                            valid_targets = [(move[2], move[3]) for move in all_moves 
                                          if move[0] == col and move[1] == row]
                
                # If a piece is already selected, attempt to move it
                else:
                    # Check if a valid move was clicked
                    if (col, row) in valid_targets:
                        # Encontrar o movimento correspondente
                        for move in valid_moves(game.board, current_player):
                            if (move[0], move[1]) == selected_piece and (move[2], move[3]) == (col, row):
                                game.apply_move(move)
                                current_player = 'B' if current_player == 'W' else 'W'
                                break
                    
                    # Reset selection regardless of whether a move was made or not
                    selected_piece = None
                    valid_targets = []

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
