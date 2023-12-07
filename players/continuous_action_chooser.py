import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

HIT_TYPES = [
    "forehand_slice",
    "forehand_topspin",
    "forehand_return",
    "forehand_serve",
    "backhand_slice",
    "backhand_topspin",
    "backhand_return",
    "backhand_volley",
]

POS_NET = 11.885
COURT_LENGTH = 23.77
COURT_WIDTH = 10.97

def main():
    # Load the data from the CSV file
    data = pd.read_csv("data/data.csv")
    clean_data = data.dropna()
    ## Deal with the position data from each side of the net
    for index in clean_data.index:
        if clean_data.loc[index, 'hitter_y'] > POS_NET:
            clean_data.loc[index, 'hitter_x'] =  COURT_WIDTH - clean_data.loc[index, 'hitter_x']
            clean_data.loc[index, 'hitter_y'] = COURT_LENGTH - clean_data.loc[index, 'hitter_y']
            clean_data.loc[index, 'receiver_x'] = COURT_WIDTH - clean_data.loc[index, 'receiver_x']
            clean_data.loc[index, 'receiver_y'] = POS_NET + (POS_NET - clean_data.loc[index, 'receiver_y'])

    # Get the state information of "ball_position"("hitter_x", "hitter_y") and "player_positions"("receiver_x", "receiver_y")
    action_info = clean_data[['hitter_x', 'hitter_y', 'receiver_x', 'receiver_y']].copy()

    # Combine "type" and "stroke" as opponent hit type
    action_info['opponent_hit_type'] = clean_data['stroke'] + "_" + clean_data['type']

    # Use "type" and "stroke" from the next line as pro player hit type
    for index in action_info.index:
        next_index = index + 1
        if next_index < len(data):
            action_info.loc[index, "player_hit_type"] = data.loc[next_index, 'stroke'] + "_" + data.loc[next_index, 'type']
            action_info.loc[index, "player_move_x"] = data.loc[next_index, "hitter_x"]
            action_info.loc[index, "player_move_y"] = data.loc[next_index, "hitter_y"]

    # Drop the rows not in HIT_TYPES
    action_info = action_info[action_info['opponent_hit_type'].isin(HIT_TYPES) & action_info['player_hit_type'].isin(HIT_TYPES)]

    # Extract features (X) and labels (y)
    ordinal_encoder = OrdinalEncoder()
    X = action_info[['hitter_x', 'hitter_y', 'receiver_x', 'receiver_y', 'opponent_hit_type']]
    X["opponent_hit_type"] = ordinal_encoder.fit_transform(X[["opponent_hit_type"]])

    label_encoder_hit_type = LabelEncoder()
    y = action_info['player_hit_type']
    y = label_encoder_hit_type.fit_transform(y)

    y_player_x = action_info['player_move_x']
    y_player_y = action_info['player_move_y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_action_x, X_test_action_x, y_train_action_x, y_test_action_x = train_test_split(X, y_player_x, test_size=0.2, random_state=42)
    X_train_action_y, X_test_action_y, y_train_action_y, y_test_action_y = train_test_split(X, y_player_y, test_size=0.2, random_state=42)

    # Create a kNN classifier with a specified number of neighbors (k)
    k = 10 # The number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier_action_x = KNeighborsRegressor(n_neighbors=k)
    knn_classifier_action_y = KNeighborsRegressor(n_neighbors=k)

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)
    knn_classifier_action_x.fit(X_train_action_x, y_train_action_x)
    knn_classifier_action_y.fit(X_train_action_y, y_train_action_y)

    # Make predictions on the test data
    predictions_hit_type = knn_classifier.predict(X_test)
    predictions_move_action_x = knn_classifier_action_x.predict(X_test_action_x)
    predictions_move_action_y = knn_classifier_action_y.predict(X_test_action_y)
    pred_hit_types = label_encoder_hit_type.inverse_transform(predictions_hit_type)

    # Evaluate the accuracy of the model
    total_position_error_x = 0
    total_position_error_y = 0
    accuracy = accuracy_score(y_test, predictions_hit_type)
    for i in range(len(predictions_move_action_x)):
        print(f"Predicted move: ({predictions_move_action_x[i]}, {predictions_move_action_y[i]})")
        print(f"Actual move: ({list(y_test_action_x)[i]}, {list(y_test_action_y)[i]})")
        total_position_error_x += predictions_move_action_x[i] - list(y_test_action_x)[i]
        total_position_error_y += predictions_move_action_y[i] - list(y_test_action_y)[i]
        print(f"Predicted hit type: {pred_hit_types[i]}")
        print(f"Actual hit type: {list(y_test)[i]}")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Manhattan Distance Difference on X: {total_position_error_x / len(predictions_move_action_x)}")
    print(f"Average Manhattan Distance Difference on Y: {total_position_error_y / len(predictions_move_action_y)}")
    # Save the encoders and model
    import pickle

    with open('model/ordinal_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(ordinal_encoder, encoder_file)

    with open('model/label_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder_hit_type, encoder_file)

    with open('model/knn_model.pkl', 'wb') as model_file:
        pickle.dump(knn_classifier, model_file)

    with open('model/knn_model_action_x.pkl', 'wb') as model_file:
        pickle.dump(knn_classifier_action_x, model_file)

    with open('model/knn_model_action_y.pkl', 'wb') as model_file:
        pickle.dump(knn_classifier_action_y, model_file)

if __name__ == "__main__":
    main()