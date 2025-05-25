Implementation of the game connect-4 using an MCTS agent

Changes to code given to us:
- check_end_state(): 
    - pass last_action to efficiently check if game has been won by the player
    - index of last action is always enough, as we check for it every turn
- implemented different playing modes (player vs. player, player vs. agent, agent vs. agent, agent vs. random agent)

Regarding move time:
- On my computer the runtime per move of the mcts agent (using default values) s approximately 2-8 seconds with early moves taking longer naturally.
  If it takes too long parameters "iterations" and the "max_depth" in the generate_move_mcts() file (mcts.py) can reduce the time it takes to make
  a move significantly. However, they might affect the performance of the agent. Currently, these parameters are not passed but would have to be 
  manually adjusted in the code. This might be improved later used **args.



AI statement:
- AI tools have been used responsibly to improve parts of this implementation, primarily for documentation, readability, and testing of the code.

