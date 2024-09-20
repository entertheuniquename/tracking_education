#!/usr/bin/python3
import random
import time

def findUnmatchedAgent(agents_in):
    for key, value in agents_in.items():
        if value is None:
            return key

def transpose(mat):
    res_mat = []
    r = len(mat)
    c = len(mat[0])
    for i in range(0, int(c)):
        temp = []
        for j in range(0, int(r)):
            temp.append(0)
        res_mat.append(temp)
    for cc in range(0, int(c)):
        for rr in range(0, int(r)):
            res_mat[cc][rr] = mat[rr][cc]
    return res_mat

def random_matrix(agents_amount, values_amount, max_value):
    values_matrix_local = []
    sequence = []

    for i in range(0, int(max_value)):
        sequence.append(i)

    for i in range(0, int(agents_amount)):
        sublist = []
        for k in range(0, int(values_amount)):
            val = random.choice(sequence)
            sublist.append(val)
        values_matrix_local.append(sublist)
    return values_matrix_local

def float_random_matrix(agents_amount, values_amount, max_value):
    values_matrix_local = []

    for i in range(0, int(agents_amount)):
        sublist = []
        for k in range(0, int(values_amount)):
            val = random.uniform(0,max_value)
            sublist.append(val)
        values_matrix_local.append(sublist)
    return values_matrix_local

#           values
#        _____________
#       | 0 1 9 6 4 8 |
#       | 4 8 2 7 2 1 |
# agents| 0 9 8 4 3 4 | <- input matrix
#       | 1 0 3 1 5 0 |
#       | 0 5 2 1 5 3 |
#
def auction_minimization_sum(input_matrix, max_value):
    print("input_matrix:")
    print(input_matrix)

    b_trans_matrix = False
    N = len(input_matrix)
    N1 = len(input_matrix[0])

    if(N>N1):
        val_matrix = transpose(input_matrix)
        N = len(val_matrix)
        N1 = len(val_matrix[0])
        b_trans_matrix = True
    else:
        val_matrix = input_matrix

    print("val_matrix:")
    print(val_matrix)

    agents_local = {}
    prices_local = {}

    for p in range(0, int(N)):
        agents_local[p] = None
    for p in range(0, int(N1)):
        prices_local[p] = 0

    i=0
    while (None in agents_local.values()):
        i=i+1
        unassigned_agent = findUnmatchedAgent(agents_local)
        agents_row = val_matrix[unassigned_agent]
        min_diff = max_value+1
        min_obj = 0
        for j in range(0, int(N1)):
            value = agents_row[j] + prices_local[j]
            if value < min_diff:
                min_diff = value
                min_obj = j

        next_min_diff = max_value+1
        for l in range(0, int(N1)):
            value = agents_row[l] - prices_local[l]
            if value < next_min_diff and value > min_diff:
                next_min_diff = value

        bid_increment = next_min_diff - min_diff
        agents_local[unassigned_agent] = min_obj

        for key, value in agents_local.items():
            if value is min_obj and key is not unassigned_agent:
                agents_local[key] = None

        prices_local[min_obj] += bid_increment

    sum = 0
    for key, value in agents_local.items():
        if(value != None):
            sum=sum+val_matrix[key][value]

    result = {}
    if(b_trans_matrix):
        for key, value in agents_local.items():
            result[value] = key
    else:
        for key, value in agents_local.items():
            result[key] = value

    return sum, result

def calculateAssignmentValue(matrix, assignment = {}):
    totalValue = 0
    for key, value in assignment.items():
        totalValue += values_matrix[key][value]
    return totalValue

def calculateAvgSolveTime():
    s = 0;
    for time in averages:
        s += time
    avgSolveTime = s / len(averages)
    return avgSolveTime

if __name__ == '__main__':
    max_value = 99.9
    vm = random_matrix(3,5,max_value)
    vm2 = float_random_matrix(5,2,max_value)
    print("values matrix:")
    print(vm2)
    [s,r] = auction_minimization_sum(vm2,max_value)
    print("minial sum: "+str(s))
    print("result pairs:"+str(r))


