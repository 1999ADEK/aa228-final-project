# aa228-final-project

### Action Chooser

#### Discrete Action Chooser
The current strategy for choosing an action is designed in `players/action_chooser.py`. It uses a k-NN that includes the information on `'hitter_area', "receiver_area", 'opponent_hit_type'` to predict `player_hit_type` and `player_move_action`.

The k-NN can be created using the dataset of the previously recorded game between Djokovic and Nadal in 2019. It should be stored in `data/data.csv`.

To get the k-NN and the associated encoders, run
```
python players/action_chooser.py
```
The resulting model and encoders should be stored under `model/`.

#### Continuous Action Chooser
The current strategy for choosing an action is designed in `players/continuous_action_chooser.py`. It uses a k-NN that includes the information on `'hitter_x', 'hitter_y', 'receiver_x', 'receiver_y', 'change_ball_dir', 'opponent_hit_type'` to predict `player_hit_type, player_move_x, player_move_y` using 3 separate kNNs.

To get the k-NNs and the associated encoders, run
```
python players/continuous_action_chooser.py
```
The resulting model and encoders should be stored under `model/`.

### Run Simulation

To simulate tennis matches, run
```bash
python main.py
  -n 100 \ # Number of matches to simulate
  -p0 default \ # Player type of player 0
  -p1 default \ # Player type of player 1
```
Please refer to  `main.py` for more details.
