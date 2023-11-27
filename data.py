import numpy as np
import pandas as pd
import math
from itertools import product



NET_Y = 11.88

CENTRAL_X = 5.485

SERVICE_LINE_Y_1 = NET_Y - 6.40

SERVICE_LINE_Y_2 = NET_Y + 6.40


STROKE_TYPES = ['forehand', 'backhand']
BALL_HIT_TYPES = ['serve', 'slice', 'topspin', 'return', 'volley', 'stop', 'smash', 'lob']
PLAYERS = ['Djokovic', 'Nadal']


def area_id(x, y):

    if y < NET_Y:
        # left service court
        if x < CENTRAL_X and y > SERVICE_LINE_Y_1:
            return 1
        # right service court 
        if x > CENTRAL_X and y > SERVICE_LINE_Y_1:
            return 2
        if x < CENTRAL_X and y < SERVICE_LINE_Y_1:
            return 3
        if x > CENTRAL_X and y < SERVICE_LINE_Y_1:
            return 4
        
    else:
        # left service court
        if x > CENTRAL_X and y < SERVICE_LINE_Y_2:
            return 5
        # right service court
        if x < CENTRAL_X and y < SERVICE_LINE_Y_2:
            return 6
        if x > CENTRAL_X and y > SERVICE_LINE_Y_2:
            return 7
        if x < CENTRAL_X and y > SERVICE_LINE_Y_2:
            return 8


def get_speed(x1, y1, x2, y2, prev_t, cur_t):
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    v = d / (cur_t - prev_t)
    return v

def get_dir(prev_x1, prev_y1, x1, y1):
    d = np.array([x1 - prev_x1, y1 - prev_y1])
    return d / np.linalg.norm(d)



if __name__ == '__main__':
    df = pd.read_csv('data/events.csv')

    # data cleaning
    for i in range(len(df)):
        if df.loc[i, 'receiver'] == '__undefined__':
            if df.loc[i, 'hitter'] == 'Djokovic':
                df.loc[i, 'receiver'] = 'Nadal'
            else:
                df.loc[i, 'receiver'] = 'Djokovic'

    # ------------------------------------------------------------------------------
    # Add area_id and ball_speed, ball_v
    # ------------------------------------------------------------------------------

    hitter_areas = []
    receiver_areas = []
    speeds = []
    velocities = [] # velocity's direction (unit vector)

    prev_t = -1
    prev_x1, prev_y1 = None, None

    for _, row in df.iterrows():
        cur_t = row['time']
        is_serve = row['isserve']
        x1, y1, x2, y2 = row['hitter_x'],  row['hitter_y'], row['receiver_x'], row['receiver_y']
        a1 = area_id(x1, y1)
        a2 = area_id(x2, y2)
        
        hitter_areas.append(a1)
        receiver_areas.append(a2)

        if prev_t != -1:
            # if it is serve, then there is a huge gap in time and we cannot calculate 
            # the speed of the previous ball hit accurately
            if is_serve:
                v = None
                d = None
            else:
                v = get_speed(prev_x1, prev_y1, x1, y1, prev_t, cur_t)
                d = get_dir(prev_x1, prev_y1, x1, y1)
            speeds.append(v)
            velocities.append(d)

        prev_t = cur_t
        prev_x1, prev_y1 = x1, y1

        
    
    # unable to calculate the speed of the last one since we only have the start time
    speeds.append(None)
    velocities.append(None)

    df['hitter_area'] = hitter_areas
    df['receiver_area'] = receiver_areas
    df['ball_speed'] = speeds
    df['ball_dir'] = velocities

    # df.to_csv('temp1.csv') # hitter area, receiver area, ball speed

    # ---------------------------------------------------------------------------------
    # Add player speed
    # ---------------------------------------------------------------------------------

    receiver_speeds = np.zeros(len(df))
    receiver_speeds.fill(np.nan)

    for i in range(1, len(df)):
        is_serve = df.loc[i, "isserve"]
        if not is_serve:
            x1, y1, prev_t = df.loc[i-1, "receiver_x"], df.loc[i-1, "receiver_y"], df.loc[i-1, "time"]
            x2, y2, cur_t = df.loc[i, "hitter_x"], df.loc[i, "hitter_y"], df.loc[i, "time"]
            v = get_speed(x1, y1, x2, y2, prev_t, cur_t)
            receiver_speeds[i-1] = v

    df["receiver_speed"] = receiver_speeds


    # ------------------------------------------------------------------------------
    # Add scores
    # ------------------------------------------------------------------------------
    df_point = pd.read_csv('data/points.csv')
    rally_id_to_score = dict(zip(df_point["rallyid"], df_point["score"]))

    df["score"] = np.nan
    current_score = "0:0, 0:0"
    for i in range(1, len(df)+1):
        prev_rally = df.loc[i-1, "rallyid"]
        curr_rally = df.loc[i, "rallyid"] if i < len(df) else df.loc[i-1, "rallyid"] + 1
        if curr_rally != prev_rally and prev_rally in rally_id_to_score:
            current_score = rally_id_to_score[prev_rally]
        df.loc[i-1, "score"] = current_score
    

    df.to_csv('data/data.csv')

    # Get ball speed data
    d_ball_speed_dict = dict((k, []) for k in product(STROKE_TYPES, BALL_HIT_TYPES))
    n_ball_speed_dict = dict((k, []) for k in product(STROKE_TYPES, BALL_HIT_TYPES))

    for i in range(len(df)):
        if not pd.isnull(df.loc[i, "ball_speed"]) and df.loc[i, "stroke"] != '__undefined__':
            v = df.loc[i, "ball_speed"]
            hitter = df.loc[i, "hitter"]
            stroke, type = df.loc[i, "stroke"], df.loc[i, "type"]

            if hitter == 'Djokovic':
                d_ball_speed_dict[(stroke, type)].append(v)
            else:
                n_ball_speed_dict[(stroke, type)].append(v)

    
    def get_ball_speed_stats(ball_speed_dict):
        for k in product(STROKE_TYPES, BALL_HIT_TYPES):
            if len(ball_speed_dict[k]) > 0:
                ball_hits = np.array(ball_speed_dict[k])
                stroke, type = k
                print(f"Stroke: {stroke}, Type: {type}, Speed: avg {np.mean(ball_hits)}, std {np.std(ball_hits)}")


    print("Djokovic: ")
    get_ball_speed_stats(d_ball_speed_dict)
    
    print("Nadal: ")
    get_ball_speed_stats(n_ball_speed_dict)
    print()


    # Get player speed for receiving each type of ball hits
    d_speed_dict = dict((k, []) for k in product(STROKE_TYPES, BALL_HIT_TYPES))
    n_speed_dict = dict((k, []) for k in product(STROKE_TYPES, BALL_HIT_TYPES))

    for i in range(len(df)):
        if not pd.isnull(df.loc[i, "receiver_speed"]) and df.loc[i, "stroke"] != '__undefined__':
            v = df.loc[i, "receiver_speed"]
            receiver = df.loc[i, "receiver"]
            stroke, type = df.loc[i, "stroke"], df.loc[i, "type"]

            if receiver == 'Djokovic':
                d_speed_dict[(stroke, type)].append(v)
            else:
                n_speed_dict[(stroke, type)].append(v)

    
    def get_player_speed_stats(player_speed_dict):
        for k in product(STROKE_TYPES, BALL_HIT_TYPES):
            if len(player_speed_dict[k]) > 0:
                speeds = np.array(player_speed_dict[k])
                stroke, type = k
                print(f"Stroke: {stroke}, Type: {type}, Speed: avg {np.mean(speeds)}, std {np.std(speeds)}")


    print("Djokovic: ")
    get_player_speed_stats(d_speed_dict)
    
    print("Nadal: ")
    get_player_speed_stats(n_speed_dict)
    print()


    # Get ball change of velocity
    d_ball_dir_dict = dict((k, []) for k in product(STROKE_TYPES, BALL_HIT_TYPES))
    n_ball_dir_dict = dict((k, []) for k in product(STROKE_TYPES, BALL_HIT_TYPES))

    for i in range(1, len(df)):
        if df.loc[i, "ball_dir"] is not None:
            isserve = df.loc[i, "isserve"]

            if isserve or df.loc[i-1, "ball_dir"].any():

                hitter_y = df.loc[i, "hitter_y"]
                hitter = df.loc[i, "hitter"]
                stroke, type = df.loc[i, "stroke"], df.loc[i, "type"]
                
                if isserve:
                    dv = df.loc[i, "ball_dir"]
                    if hitter_y > NET_Y:
                        dv *= -1.0
                    
                    dtheta = np.arctan2(dv[1], dv[0])
                    if hitter == "Nadal":
                        n_ball_dir_dict[(stroke, type)].append(dtheta)
                    else:
                        d_ball_dir_dict[(stroke, type)].append(dtheta)

                else:
                    if hitter_y < NET_Y:
                        v0 = df.loc[i-1, "ball_dir"]
                        v1 = df.loc[i, "ball_dir"]
                    else:
                        v0 = - df.loc[i-1, "ball_dir"]
                        v1 = - df.loc[i, "ball_dir"]
                    
                    theta0 = np.arctan2(v0[1], v0[0])
                    theta1 = np.arctan2(v1[1], v0[0])

                    dtheta = theta1 - theta0
                    if hitter == "Nadal":
                        n_ball_dir_dict[(stroke, type)].append(dtheta)
                    else:
                        d_ball_dir_dict[(stroke, type)].append(dtheta)

    
    def get_ball_change_dir_stats(ball_dir_dict):
        for k in product(STROKE_TYPES, BALL_HIT_TYPES):
            if len(ball_dir_dict[k]) > 0:
                ball_dirs = np.array(ball_dir_dict[k])
                stroke, type = k
                print(f"Stroke: {stroke}, Type: {type}, Change of dir: avg {np.rad2deg(np.mean(ball_dirs))}, std {np.rad2deg(np.std(ball_dirs))}")


    print("Djokovic: ")
    get_ball_change_dir_stats(d_ball_dir_dict)
    
    print("Nadal: ")
    get_ball_change_dir_stats(n_ball_dir_dict)         
