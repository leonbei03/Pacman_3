# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"
        
        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        # Base score
        score = successor_game_state.get_score()

        # Food Proximity Reward
        food_list = new_food.as_list()
        if food_list:
            # We take the nearest food since is the one that we want to eat next
            food_distances = [util.manhattan_distance(new_pos, food) for food in food_list]
            min_food_distance = min(food_distances)
            score += 10 / (min_food_distance + 1)

        # Ghost Proximity Penalty
        for ghost_state, scared_time in zip(new_ghost_states, new_scared_times):
            ghost_pos = ghost_state.get_position()
            ghost_distance = util.manhattan_distance(new_pos, ghost_pos)

            if scared_time > 0:
                # Reward moving closer to edible ghosts (to chase them)
                score += 30 / (ghost_distance + 1)
            else:
                # Apply penalties for proximity to active ghosts
                if ghost_distance < 2:
                    score -= 200  # High penalty for very close ghosts
                elif ghost_distance < 4:
                    score -= 50 / (ghost_distance + 1)  # Smaller penalty for moderately close ghosts

        # Small penalty for each remaining food item to encourage clearing food
        score -= 2 * len(food_list)

        # Penalty for STOP action High, discourages inaction
        if action == Directions.STOP:
            score -= 50 

        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Get the number of pacman + ghosts
        num_agents = game_state.get_num_agents()
        # Calculate the maximum depth based on the number of ghosts          
        max_depth = num_agents * self.depth
        
        # Maximize the value for Pacman, recursively considering the ghost's moves
        def max_value(state, depth, agent_index, num_agents):
            # Return the evaluation of the current state if we have reached a terminal state or max depth since there is no point on keep exploring
            if state.is_win() or state.is_lose() or depth == max_depth:
                return self.evaluation_function(state)
            
            max_eval = float('-inf')
            legal_actions = state.get_legal_actions(agent_index)
            
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                eval = minimax(successor, depth + 1, (agent_index + 1) % num_agents)
                if eval > max_eval:
                    max_eval = eval
                    if depth == 0:  # If it's the root level, set it as the best action
                        best_action = action
                        
            return best_action if depth == 0 else max_eval

        # Minimize the value for ghosts, considering Pacman's possible actions
        def min_value(state, depth, agent_index, num_agents):
            # Return the evaluation of the current state if we have reached a terminal state or max depth since there is no point on keep exploring
            if state.is_win() or state.is_lose() or depth == max_depth:
                return self.evaluation_function(state)
            
            min_eval = float('inf')
            legal_actions = state.get_legal_actions(agent_index)

            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                eval = minimax(successor, depth + 1, (agent_index + 1) % num_agents)
                min_eval = min(min_eval, eval)
                
            return min_eval

        # Start the minimax algorithm from Pacman's turn (index 0)
        def minimax(state, depth, agent_index):
            
            # If we've reached the maximum depth or a terminal state, evaluate the state 
            if depth == max_depth or state.is_win() or state.is_lose():
                return self.evaluation_function(state)

            # If it's Pacman's turn (maximizing player)
            if agent_index == 0:
                return max_value(state, depth, agent_index, num_agents)
            else:
                # If it's a ghost's turn (minimizing player)
                return min_value(state, depth, agent_index, num_agents)
        
        # Start the minimax search with Pacman (agent_index=0) at depth 0
        return minimax(game_state, 0, 0)    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
