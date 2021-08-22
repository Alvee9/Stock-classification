
def cust_dist(a, b):
    d = 0.0
    for i in range(1, min(len(a), len(b))):
        if ((a[i] - a[i - 1]) * (b[i] - b[i - 1]) < 0):
            d += abs(a[i] - b[i]) * 4
        else:
            change_diff = abs((a[i] - a[i - 1]) - (b[i] - b[i - 1]))
            d += change_diff
        if i >= 7:
            if ((a[i] - a[i - 7]) * (b[i] - b[i - 7]) < 0):
                d += abs(a[i] - a[i - 7] - (b[i] - b[i - 7])) * 8
            else:
                change_diff = abs((a[i] - a[i - 7]) - (b[i] - b[i - 7]))
                d += change_diff * 2
        if i >= 30:
            if ((a[i] - a[i - 30]) * (b[i] - b[i - 30]) < 0):
                d += abs(a[i] - a[i - 30] - (b[i] - b[i - 30])) * 12
            else:
                change_diff = abs((a[i] - a[i - 30]) - (b[i] - b[i - 30]))
                d += change_diff * 3

        d += abs(a[i] - b[i])/2
    
    return d


def get_dist_matrix(data):
    distance_matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            row.append(cust_dist(data[i][1:], data[j][1:]))
        distance_matrix.append(row)
    return distance_matrix