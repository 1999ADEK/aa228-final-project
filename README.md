# aa228-final-project

### Action Chooser
The current strategy for choosing an action is designed in `players/action_chooser.py`. It uses a k-NN that includes the information on `'hitter_area', "receiver_area", 'opponent_hit_type'` to predict `player_hit_type`.

The k-NN can be created using the dataset of the previously recorded game between Djokovic and Nadal in 2019. It should be stored in `data/data.csv`.

To get the k-NN and the associated encoders, run
```
python players/action_chooser.py
```
The resulting model and encoders should be stored under `model/`.

### Simulate a Match
**TODO:** Add description ...?

```
python simulate.py
```