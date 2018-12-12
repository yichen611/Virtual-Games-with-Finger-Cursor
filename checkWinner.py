

def check(order):

    result = -1
    winner = 0
   
    if order[0] == order[1] and order[0] == order[2] and order[0] != -1:
        result = 1
        winner = order[0]
    elif order[0] == order[3] and order[0] == order[6] and order[0] != -1:
        result = 1
        winner = order[0]
    elif order[0] == order[4] and order[0] == order[8] and order[0] != -1:
        result = 1
        winner = order[0]
    elif order[1] == order[4] and order[1] == order[7] and order[0] != -1:
        result = 1
        winner = order[1]
    elif order[2] == order[5] and order[2] == order[8] and order[2] != -1:
        result = 1
        winner = order[2]
    elif order[2] == order[4] and order[2] == order[6] and order[2] != -1:
        result = 1
        winner = order[2]
    elif order[3] == order[4] and order[3] == order[5] and order[3] != -1:
        result = 1
        winner = order[3]
    elif order[6] == order[7] and order[6] == order[8] and order[6] != -1:
        result = 1
        winner = order[6]

    if result == 1:
        if winner == 1:
            print("O wins!")
        else:
            print("X wins!")
    else:
        result = 0
        for item in order:
            if item == -1:
                result = -1
                print("continue")
                break

        print("DRAW")

    return result, winner;



