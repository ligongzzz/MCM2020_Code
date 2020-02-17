# The passing network in a play.
import numpy as np
import cv2
import csv
import AHP_utils

# Hyper parameters.
is_huskies: bool = True
matchid = 1

# Read the match Result file.
match_file = list(csv.reader(open('./data/matches.csv')))
action_file = list(csv.reader(open('./data/fullevents.csv')))[1:]
csv_reader = list(csv.reader(open('./data/passingevents.csv')))[1:]

# Storage all the data.
offensive_cnt = np.zeros(76)


def get_storage_id(matchid, is_huskies):
    '''
    Transform the match ID and is_huskies to the storage_id.
    '''
    if is_huskies:
        return (matchid - 1) * 2
    else:
        return matchid * 2 - 1


def get_offensive_id(storage_id, index):
    '''
    Return the offensive id in the offensive vector.
    '''
    cur_cnt = 0
    for i in range(storage_id):
        cur_cnt += offensive_cnt[i]
    return cur_cnt + index


while matchid <= 38:
    passing_list = None
    # Loading the matching data.

    if is_huskies:
        passing_list = [row for row in csv_reader if row[0]
                        == str(matchid) and row[1] == 'Huskies']
        action_list = [row for row in action_file if row[0] == str(matchid)]
    else:
        passing_list = [row for row in csv_reader if row[0]
                        == str(matchid) and row[1] != 'Huskies']
        action_list = [row for row in action_file if row[0] == str(matchid)]
    passing_cnt = len(passing_list)

    # Analyzing the data.
    player_map = {}
    player_list = []
    for row in passing_list:
        if player_map.get(row[2]) is None:
            player_map[row[2]] = len(player_list)
            player_list.append(row[2])
        if player_map.get(row[3]) is None:
            player_map[row[3]] = len(player_list)
            player_list.append(row[3])

    player_cnt = len(player_list)
    w: np.ndarray = np.zeros((player_cnt, player_cnt))
    player_pos = np.zeros((player_cnt, 2))
    player_passing_cnt = np.zeros(player_cnt)

    # Count the passing cnt.
    for row in passing_list:
        origin_id = player_map[row[2]]
        dest_id = player_map[row[3]]

        # Update the passing matrix.
        if origin_id != dest_id:
            w[origin_id][dest_id] += 1

        # Update the player's position.
        player_pos[origin_id][0] += float(row[7])
        player_pos[origin_id][1] += float(row[8])
        player_pos[dest_id][0] += float(row[9])
        player_pos[dest_id][1] += float(row[10])

        player_passing_cnt[origin_id] += 1
        player_passing_cnt[dest_id] += 1

    for pos in range(player_cnt):
        if player_passing_cnt[pos] != 0:
            player_pos[pos][0] /= player_passing_cnt[pos]
            player_pos[pos][1] /= player_passing_cnt[pos]

    # Calculate the clustering coefficient.
    cw_list = []
    for i in range(player_cnt):
        cw1 = 0
        cw2 = 0
        for j in range(player_cnt):
            for k in range(player_cnt):
                cw1 += w[i, j] * w[j, k] * w[i, k]
                cw2 += w[i, j] * w[i, k]
        cw_list.append(cw1 / cw2 if cw2 != 0 else 0)
    C_ans = np.mean(np.array(cw_list))

    # Calculate the shortest path.
    d = 1.0 / (w + 1e-9)
    for k in range(player_cnt):
        for i in range(player_cnt):
            for j in range(player_cnt):
                if d[i, j] > d[i, k] + d[k, j]:
                    d[i, j] = d[i, k] + d[k, j]

    # Remove the dirty data.
    d_cnt = 0
    d_sum = 0.0
    for i in range(player_cnt):
        for j in range(player_cnt):
            if i != j and d[i, j] < 1e5:
                d_sum += d[i, j]
                d_cnt += 1

    D_ans = d_sum / d_cnt

    # Calculate the largest eigenvalue of the adjacency matrix.
    L1_ans = np.max(np.linalg.eig(w)[0].real)

    # Calculate the second smallest eigenvalue of the Laplacian matrix L = S - A.
    S_matrix = np.diag([np.sum(w[i]) for i in range(player_cnt)])
    L_matrix = S_matrix - w
    L_val: list = np.linalg.eig(L_matrix)[0].real.tolist()
    L_val.sort()
    L2_ans = L_val[1]

    # Analyze the offensive tactics.
    for eventid in range(len(action_list)):
        cur_event = action_list[eventid]
        if cur_event[7] != 'Shot':
            continue

        # Shot, analyze the actions before and after it.
        event_before = action_list[eventid -
                                   21: eventid] if eventid - 21 >= 0 else action_list[:eventid]

        event_after = action_list[eventid +
                                  1] if eventid + 1 < len(action_list) else None

        # Caculate the time.
        total_time = float(action_list[eventid][5]) - float(event_before[0][5])
        ball_time = 0.0
        num1 = 0
        num2 = 0
        num3 = 0
        simplepass = 0
        smartpass = 0
        flag = 0

        for i, event in enumerate(event_before):
            if i+1 < len(event_before) and ((is_huskies and event[1] == 'Huskies') or (not is_huskies and event[1] != 'Huskies')):
                ball_time += float(event_before[i + 1][5]) - float(event[5])
                if event[7] == 'Ground attacking duel' and i+2 < len(event_before) and event_before[i + 2][6] == 'Pass':
                    num1 += 1
                elif event[7] == 'Ground attacking duel' and i+2 < len(event_before) and event_before[i + 2][6] == 'Duel':
                    num2 += 1
                elif event[7] == 'Simple pass':
                    simplepass += 1
                elif event[7] == 'Smart pass':
                    smartpass += 1
                elif event[7] == 'Corner':
                    flag = 1
            elif event[6] == 'Foul' and i + 1 < len(event_before) and event_before[i + 1][6] == 'Free Kick':
                num3 += 1
        time_rate = ball_time / total_time

        num5 = 1 if event_after is not None and event_after[6] == 'Save attempt' else 0

    # Analyze the defensive tactics.
    ODC = 0
    PPDA_cnt = 0
    PPDA_action = 0
    shot_cnt = 0
    goal_cnt = 0
    clear_cnt = 0
    save_cnt = 0

    for action in action_list:
        if (is_huskies and action[1] != 'Huskies') or (not is_huskies and action[1] == 'Huskies'):
            if action[6] == 'Pass' and float(action[8]) > 82.6 and action[3] != '':
                ODC += 1
            if action[6] == 'Pass' and float(action[8]) > 62 and action[3] != '':
                PPDA_cnt += 1
        else:
            if action[6] == 'Duel' or action[6] == 'Save attempt' or action[6] == 'Foul':
                PPDA_action += 1
            elif action[7] == 'Clearance':
                clear_cnt += 1
            elif action[6] == 'Save attempt':
                save_cnt += 1
            elif action[6] == 'Shot':
                shot_cnt += 1
    if is_huskies:
        goal_cnt = int(match_file[matchid][3])
    else:
        goal_cnt = int(match_file[matchid][4])
    PPDA = PPDA_cnt / PPDA_action
    defence_val = 1 / ODC + 1 / PPDA - goal_cnt / shot_cnt
    # Print the ans.
    print(
        f'MatchID:{matchid} Huskies:{is_huskies} C:{C_ans} D:{D_ans} L1:{L1_ans} L2:{L2_ans} '
        f'HuskiesResult:{match_file[matchid][2]}')

    # Continue.
    if is_huskies:
        is_huskies = False
    else:
        is_huskies = True
        matchid += 1
