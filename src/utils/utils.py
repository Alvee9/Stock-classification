# Returns the distance between vectors a and b
def cust_dist(a, b):
    d = 0.0
    # Iterate over all the elements in the vectors starting from the second one
    for i in range(1, min(len(a), len(b))):
        # Check if the daily trends of the stocks similar or different
        if ((a[i] - a[i - 1]) * (b[i] - b[i - 1]) < 0):
            # The daily trends are different, it incurs a bigger penalty
            d += abs(a[i] - b[i]) * 4
        else:
            # The daily trends are similar, it incurs a smaller penalty
            change_diff = abs((a[i] - a[i - 1]) - (b[i] - b[i - 1]))
            d += change_diff

        if i >= 7:
            # Check if the weekly trends of the stocks similar or different
            if ((a[i] - a[i - 7]) * (b[i] - b[i - 7]) < 0):
                # The weekly trends are different, it incurs a bigger penalty
                d += abs(a[i] - a[i - 7] - (b[i] - b[i - 7])) * 8
            else:
                # The weekly trends are similar, it incurs a smaller penalty
                change_diff = abs((a[i] - a[i - 7]) - (b[i] - b[i - 7]))
                d += change_diff * 2
        
        if i >= 30:
            # Check if the monthly trends of the stocks similar or different
            if ((a[i] - a[i - 30]) * (b[i] - b[i - 30]) < 0):
                # The monthly trends are different, it incurs a bigger penalty
                d += abs(a[i] - a[i - 30] - (b[i] - b[i - 30])) * 12
            else:
                # The monthly trends are similar, it incurs a smaller penalty
                change_diff = abs((a[i] - a[i - 30]) - (b[i] - b[i - 30]))
                d += change_diff * 3

        # Finally the absolute difference between two corresponding elements are halved and added as it is considered as a small penalty
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

def get_edge_matrix(data):
    distance_matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            row.append(1 / (1 + cust_dist(data[i][1:], data[j][1:])))
        distance_matrix.append(row)
    return distance_matrix
