import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import game

class CNN_DuelingDQN(nn.Module):
    """The Dueling DQN architecture for better state-value estimation."""
    def __init__(self, in_channels, action_dim):
        super(CNN_DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2), nn.ReLU()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 4, 4)
            conv_out_size = self.conv(dummy_input).flatten().shape[0]

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class PlayerAgent:
    def __init__(self, model_path, device, depth=3):
        self.device = device
        self.depth = depth
        print(f"Loading model from: {model_path}")
        self.model = CNN_DuelingDQN(in_channels=1, action_dim=4).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['q_net'])
            self.model.eval()
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    def _get_obs(self, board):
        """Converts the board to the log2 format the model expects."""
        with np.errstate(divide='ignore'):
            return np.where(board > 0, np.log2(board), 0.0).astype(np.float32)

    def _evaluate_board_with_dqn(self, board):
        """Evaluates a board state using the model's value stream for a single score."""
        with torch.no_grad():
            obs = self._get_obs(board)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            features = self.model.conv(state_tensor)
            features = features.view(features.size(0), -1)
            value = self.model.value_stream(features).item()
            return value

    def _get_valid_moves(self, board):
        """Returns a boolean array of valid moves."""
        valid_moves = [False] * 4 
        for i, direction in enumerate(['up', 'right', 'down', 'left']):
            next_board, _ = game.move_board(np.copy(board), direction)
            if not np.array_equal(board, next_board):
                valid_moves[i] = True
        return np.array(valid_moves)
    
    def select_move_ai_only(self, board):
        """Selects a move using only the raw DQN model output."""
        valid_moves = self._get_valid_moves(board)
        if not np.any(valid_moves):
            return -1 

        with torch.no_grad():
            obs = self._get_obs(board)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            q_values[0, ~valid_moves] = -float('inf')
            best_action = torch.argmax(q_values).item()
        return best_action

    def select_move_with_alg(self, board):
        """Selects a move using the ExpectiMax algorithm, guided by the DQN."""
        valid_moves = self._get_valid_moves(board)
        if not np.any(valid_moves):
            return -1
            
        _, best_action = self.expectimax(board, self.depth, True)
        
        if best_action is None:
            print("ExpectiMax failed to find a move, falling back to AI-only.")
            return self.select_move_ai_only(board)
            
        return best_action

    def expectimax(self, board, depth, is_player_turn=True):
        """The core ExpectiMax search algorithm."""
        if depth == 0 or game.is_game_over(board):
            return self._evaluate_board_with_dqn(board), None

        if is_player_turn:
            max_score = -float('inf')
            best_action = None
            valid_moves = self._get_valid_moves(board)
            
            for action in range(4):
                if valid_moves[action]:
                    direction = ['up', 'right', 'down', 'left'][action]
                    new_board, _ = game.move_board(np.copy(board), direction)
                    score, _ = self.expectimax(new_board, depth - 1, False)
                    if score > max_score:
                        max_score = score
                        best_action = action
            return max_score, best_action
        
        else:
            empty_cells = list(zip(*np.where(board == 0)))
            if not empty_cells:
                return self._evaluate_board_with_dqn(board), None

            expected_score = 0
            for r, c in empty_cells:
                board_with_2 = board.copy(); board_with_2[r, c] = 2
                score2, _ = self.expectimax(board_with_2, depth - 1, True)
                expected_score += 0.9 * score2

                board_with_4 = board.copy(); board_with_4[r, c] = 4
                score4, _ = self.expectimax(board_with_4, depth - 1, True)
                expected_score += 0.1 * score4
                
            return expected_score / len(empty_cells), None


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = "2048_best_new.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
try:
    agent = PlayerAgent(model_path=MODEL_PATH, device=device, depth=2)
    print("Agent initialized successfully!")
except Exception as e:
    print(f"FATAL: Error initializing agent: {e}")
    agent = None

@app.route('/get-move-ai', methods=['POST'])
def get_move_ai():
    """Endpoint for getting a move from the AI without ExpectiMax."""
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 500
    try:
        board_np = np.array(request.get_json()['board'], dtype=int)
        best_move = agent.select_move_ai_only(board_np)
        return jsonify({'move': int(best_move)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/get-move-ai-alg', methods=['POST'])
def get_move_ai_alg():
    """Endpoint for getting a move from the AI with ExpectiMax search."""
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 500
    try:
        board_np = np.array(request.get_json()['board'], dtype=int)
        best_move = agent.select_move_with_alg(board_np)
        return jsonify({'move': int(best_move)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)