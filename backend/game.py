import numpy as np
import random

BOARD_SIZE = 4

def get_empty_board():
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

def add_random_tile(board):
    new_board = np.copy(board)
    empty_cells = list(zip(*np.where(new_board == 0)))
    
    if not empty_cells:
        return new_board
    
    i, j = random.choice(empty_cells)
    new_board[i, j] = 2 if random.random() < 0.9 else 4
    return new_board

def slide_and_combine(row):
    new_row = row[row != 0] 
    score_gained = 0
    result_row = []
    
    i = 0
    while i < len(new_row):
        if i + 1 < len(new_row) and new_row[i] == new_row[i+1]:
            merged_value = new_row[i] * 2
            result_row.append(merged_value)
            score_gained += merged_value
            i += 2
        else:
            result_row.append(new_row[i])
            i += 1
            
    while len(result_row) < BOARD_SIZE:
        result_row.append(0)
        
    return np.array(result_row), score_gained

def move_board(board, direction):
    rotated_board = np.copy(board)
    
    if direction == 'up':
        rotated_board = np.rot90(rotated_board)
    elif direction == 'right':
        rotated_board = np.rot90(rotated_board, 2)
    elif direction == 'down':
        rotated_board = np.rot90(rotated_board, 3)

    new_board = np.zeros_like(rotated_board)
    total_score_gained = 0
    
    for i, row in enumerate(rotated_board):
        new_row, score = slide_and_combine(row)
        new_board[i] = new_row
        total_score_gained += score
        
    if direction == 'up':
        new_board = np.rot90(new_board, -1)
    elif direction == 'right':
        new_board = np.rot90(new_board, -2)
    elif direction == 'down':
        new_board = np.rot90(new_board, -3)
        
    return new_board, total_score_gained

def is_game_over(board):
    """Checks if the game is over (no empty cells and no possible merges)."""
    if 0 in board:
        return False
        
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            current = board[i, j]
            if j < BOARD_SIZE - 1 and current == board[i, j + 1]:
                return False
            if i < BOARD_SIZE - 1 and current == board[i + 1, j]:
                return False
                
    return True