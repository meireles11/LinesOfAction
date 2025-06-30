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
    w1, w2, w3 = 1, 6, 6
    
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
def is_winner(state, player):
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
GREEN = (0,255,0)
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
        self.player = 'B' if self.player == 'W' else 'W'  # Troca de jogador
    
    def reset(self):
        self.board = self.initialize_board()
        self.player = 'B'



def apply_move_copy(state, move):
    """Aplica um movimento ao estado e retorna um novo estado sem modificar o original e sem trocar de jogador."""
    new_state = deepcopy(state)  # Criar uma cópia do estado
    xi, yi, xf, yf = move
    new_state.board[yf][xf] = new_state.board[yi][xi]
    new_state.board[yi][xi] = '.'
    return new_state

def minimax(state, depth, maximizing_player, alpha, beta, original_player):
    """Implementação do algoritmo Minimax com poda alfa-beta."""
    winner = check_victory(state)
    if winner:
        if winner == state.player:
            return 10000
        else:
            return -10000
    if depth == 0:
        return combined_evaluation(state.board, original_player)
    
    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves(state.board, state.player):
            new_state = apply_move_copy(state, move)
            new_state.player = 'B' if new_state.player == 'W' else 'W'
            evaluation = minimax(new_state, depth - 1, False, alpha, beta, original_player)
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in valid_moves(state.board, state.player):
            new_state = apply_move_copy(state, move)
            new_state.player = 'B' if new_state.player == 'W' else 'W'
            evaluation = minimax(new_state, depth - 1, True, alpha, beta, original_player)
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval

def best_move(state, depth):
    """Retorna o melhor movimento para o jogador atual usando Minimax."""
    best_move = None
    alpha, beta = -math.inf, math.inf
    maximizing = state.player == True
    original_player = state.player
    best_score = -math.inf if maximizing else math.inf

    for move in valid_moves(state.board, state.player):
        new_state = apply_move_copy(state, move)
        new_state.player = 'B' if new_state.player == 'W' else 'W'
        score = minimax(new_state, depth - 1, not maximizing, alpha, beta, original_player)

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
    """
    Implementação simplificada do Monte Carlo Tree Search (MCTS).
    Faz simulações aleatórias e escolhe a jogada que levou a mais vitórias.
    """
    move_scores = {move: 0 for move in valid_moves(state.board, state.player)}

    for _ in range(simulations):
        move = random.choice(list(move_scores.keys()))
        new_state = apply_move_copy(state, move)
        new_state.player = 'B' if new_state.player == 'W' else 'W'
        result = simulate_game(new_state)
        move_scores[move] += result

    # Escolhe a jogada com maior score nas simulações
    best_move = max(move_scores, key=move_scores.get)
    return best_move

def simulate_game(state):
    """
    Simula um jogo aleatório a partir do estado atual e retorna:
    - 1 se o jogador atual ganhar
    - -1 se o oponente ganhar
    - 0 se der empate
    """
    player = state.player
    while True:
        moves = valid_moves(state.board, player)
        if not moves:
            return 0  # Empate

        move = random.choice(moves)
        state = apply_move_copy(state, move)

        # Verifica se há um vencedor após a jogada
        winner = check_victory(state)
        if winner is not None:
            return 1 if winner == player else -1  # Quem fez a última jogada ganha

        player = 'B' if player == 'W' else 'W'  # Alterna o jogador

def strategy(state, depth=3):
    """
    Escolhe se usa Minimax ou Monte Carlo para determinar a melhor jogada.
    Usa Minimax para estados iniciais (poucas jogadas) e MCTS para estados mais complexos.
    """
    num_moves = len(valid_moves(state.board, state.player))

    if num_moves <= 1000:  # Número arbitrário, pode ser ajustado conforme testes
        return best_move(state, depth)  # Usa Minimax
    else:
        return mcts_search(state, simulations=1000)  # Usa MCTS quando há muitas opções


random_ai_delay = 3000
def random_strategy(state):
        moves = valid_moves(state.board, state.player)
        pygame.time.delay(random_ai_delay)  # Delay específico para AI aleatória
        return random.choice(moves) if moves else None

def main():
    pygame.init()
    WIDTH = TILE_SIZE * BOARD_SIZE
    HEIGHT = TILE_SIZE * BOARD_SIZE
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Lines of Action - AI vs AI")

    game = LineOfAction()
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)
    button_font = pygame.font.SysFont('Arial', 24)

    # Configurações das AIs
    AI_CONFIG = {
        'B': {'type': 'minimax', 'depth': 1},  # Pretas: AI forte
        'W': {'type': 'minimax', 'depth': 2}   # Brancas: AI aleatória
    }

    # Variáveis de controle
    running = True
    winner = None
    game_speed = 20
    ai_delay = 200
    last_move_time = 0
    paused = False
    last_board = None

    # Função para desenhar tudo
    def draw_everything():
        surface = pygame.Surface((WIDTH, HEIGHT))
        surface.fill(BROWN)
        
        # Tabuleiro
        for x in range(0, WIDTH, TILE_SIZE):
            pygame.draw.line(surface, BLACK, (x, 0), (x, HEIGHT), 2)
        for y in range(0, HEIGHT, TILE_SIZE):
            pygame.draw.line(surface, BLACK, (0, y), (WIDTH, y), 2)
        
        # Peças
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                piece = game.board[y][x]
                if piece != '.':
                    pygame.draw.circle(surface, PIECES[piece],
                                     (x*TILE_SIZE + TILE_SIZE//2, y*TILE_SIZE + TILE_SIZE//2),
                                     TILE_SIZE//3)
        
        # UI
        info_texts = [
            f"Jogador: {'Pretas (AI Forte)' if game.player == 'B' else 'Brancas (AI Aleatória)'}",
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, WHITE)
            surface.blit(text_surface, (10, 10 + i*30))
        
        if winner:
            message = "Brancas venceram!" if winner == 'W' else "Pretas venceram!"
            text_surface = font.render(message, True, GREEN)
            text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2 - 20))
            surface.blit(text_surface, text_rect)
            
            restart_text = button_font.render("Pressione R para reiniciar", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 20))
            surface.blit(restart_text, restart_rect)
        
        return surface

    # Desenhar estado inicial
    last_board = draw_everything()
    screen.blit(last_board, (0, 0))
    pygame.display.flip()

    def process_ai_move():
        nonlocal game, winner, last_move_time, last_board, paused
        current_time = pygame.time.get_ticks()
        
        if not paused and current_time - last_move_time > ai_delay // game_speed:
            ai_config = AI_CONFIG[game.player]
            
            if ai_config['type'] == 'minimax':
                move = strategy(game, depth=ai_config['depth'])
            elif ai_config['type'] == 'random':
                move = random_strategy(game)  # Esta função agora tem delay incorporado
            else:
                move = strategy(game, depth=1)
            
            if move:
                game.apply_move(move)
                last_move_time = current_time
                winner = check_victory(game)
                if winner:    
                    paused = True
                last_board = draw_everything()
                return True
        return False

    # Loop principal
    while running:
        current_time = pygame.time.get_ticks()
        
        # Processar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    last_board = draw_everything()
                elif event.key == pygame.K_UP:
                    game_speed = min(game_speed + 1, 20)
                    last_board = draw_everything()
                elif event.key == pygame.K_DOWN:
                    game_speed = max(game_speed - 1, 1)
                    last_board = draw_everything()
                elif event.key == pygame.K_n and paused:
                    if process_ai_move():
                        screen.blit(last_board, (0, 0))
                        pygame.display.flip()
                elif event.key == pygame.K_r:
                    game.reset()
                    winner = None
                    last_move_time = current_time
                    last_board = draw_everything()
                    screen.blit(last_board, (0, 0))
                    pygame.display.flip()

        # Processar movimento da AI
        if not paused:
            if process_ai_move():
                screen.blit(last_board, (0, 0))
                pygame.display.flip()

        # Redesenhar continuamente
        screen.blit(last_board, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

def random_move(state):
    moves = valid_moves(state.board, state.player)
    if moves:
        return random.choice(moves)
    return None


# Jogo com 100 simulações e estatísticas
def simulate_games(num_games=100):
    results = {"B": 0, "W": 0, "Draw": 0}
    
    for _ in range(num_games):
        game = LineOfAction()  # Inicia um novo jogo
        while True:
            winner = check_victory(game)
            if winner:
                if winner == "B":
                    results["B"] += 1
                    print(results)
                elif winner == "W":
                    results["W"] += 1
                    print(results)
                else:
                    results["Draw"] += 1
                    print(results)
                break

            # Jogada do jogador "B" (IA)
            if game.player == 'B':
                move = strategy(game, depth=3)
                  # IA faz a jogada
            # Jogada do jogador "W" (aleatório)
            else:
                move = strategy(game, depth=1)

            if move:
                game.apply_move(move)
            else:
                results["Draw"] += 1
                break  # Empate se não houver jogadas válidas

    # Exibindo as estatísticas após os 100 jogos
    print(f"Resultados após {num_games} jogos:")
    print(f"Vitórias do jogador 'B': {results['B']}")
    print(f"Vitórias do jogador 'W': {results['W']}")
    print(f"Empates: {results['Draw']}")

# Chamar a função para simular 100 jogos

simulate_games(100)