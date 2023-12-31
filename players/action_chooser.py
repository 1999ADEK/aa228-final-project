import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

position_table = {
    1: "TL",
    2: "TR",
    3: "BL",
    4: "BR",
    5: "TL",
    6: "TR",
    7: "BL",
    8: "BR"
}

def main():
    # Load the data from the CSV file
    data = pd.read_csv("data/data.csv")
    clean_data = data.dropna()
    # Change number position to string position from each player's perspective
    clean_data[['hitter_area', 'receiver_area']] = clean_data[['hitter_area', 'receiver_area']].replace(position_table)

    # Get the state information of "ball_position"("hitter_area") and "player_positions"("receiver_area")
    action_info = clean_data[['hitter_area', "receiver_area"]].copy()
    # Combine "type" and "stroke" as opponent hit type
    action_info['opponent_hit_type'] = clean_data['stroke'] + "_" + clean_data['type']

    # Use "type" and "stroke" from the next line as pro player hit type
    for index in action_info.index:
        next_index = index + 1
        if next_index < len(data):
            action_info.loc[index, "player_hit_type"] = data.loc[next_index, 'stroke'] + "_" + data.loc[next_index, 'type']
            action_info.loc[index, "player_move_action"] = data.loc[next_index, "hitter_area"]

    # Drop the rows not in HIT_TYPES
    action_info = action_info[action_info['opponent_hit_type'].isin(HIT_TYPES) & action_info['player_hit_type'].isin(HIT_TYPES)]

    # Extract features (X) and labels (y)
    ordinal_encoder = OrdinalEncoder()
    X = action_info[['hitter_area', "receiver_area", 'opponent_hit_type']]
    X = ordinal_encoder.fit_transform(X)

    label_encoder_hit_type = LabelEncoder()
    y = action_info['player_hit_type']
    y = label_encoder_hit_type.fit_transform(y)

    label_encoder_move_action = LabelEncoder()
    y_action = action_info['player_move_action']
    y_action = label_encoder_move_action.fit_transform(y_action)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_action, X_test_action, y_train_action, y_test_action = train_test_split(X, y_action, test_size=0.2, random_state=42)

    # Create a kNN classifier with a specified number of neighbors (k)
    k = 10 # The number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier_action = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)
    knn_classifier_action.fit(X_train_action, y_train_action)

    # Make predictions on the test data
    predictions_hit_type = knn_classifier.predict(X_test)
    predictions_move_action = knn_classifier_action.predict(X_test_action)
    pred_hit_types = label_encoder_hit_type.inverse_transform(predictions_hit_type)
    pred_move_actions = label_encoder_move_action.inverse_transform(predictions_move_action)
    print(predictions_hit_type, y_test)
    print(predictions_move_action, y_test_action)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions_hit_type)
    accuracy_action = accuracy_score(y_test_action, predictions_move_action)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Accuracy of action: {accuracy_action * 100:.2f}%")

    # Save the encoders and model
    import pickle

    with open('model/ordinal_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(ordinal_encoder, encoder_file)

    with open('model/label_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder_hit_type, encoder_file)

    with open('model/label_encoder_action.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder_move_action, encoder_file)

    with open('model/knn_model.pkl', 'wb') as model_file:
        pickle.dump(knn_classifier, model_file)

    with open('model/knn_model_action.pkl', 'wb') as model_file:
        pickle.dump(knn_classifier_action, model_file)
if __name__ == "__main__":
    main()