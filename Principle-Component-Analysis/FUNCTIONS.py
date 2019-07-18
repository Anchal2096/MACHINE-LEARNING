x = [1, -1, 4]
y = [2, 1, 3]
z = [1, 3, -1]


def cov(y, z):
    if len(x) != len(y):
        print("length of two lists are not equal")
        return

    n = len(x)
    """ 
    xy =[]
    for i in range(10):
        prod = x[i] * y[i]
        xy.append(prod)
    print(xy)
    """
    xy = [x[i] * y[i] for i in range(n)]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    return (sum(xy) - n * mean_x * mean_y) / (n - 1)


print('Covariance : ' + str(cov(x, y)))


def sd(x):
    if len(x) == 0:
        return 0
    n = len(x)

    mean_x = sum(x) / n
    variance = sum([(x[i] - mean_x) ** 2 for i in range(n)]) / float(n)
    return variance ** 0.5


print('Standard Deviation : ', cov(x, y))


def corr(x, y):
    if len(x) != len(y):
        return

    correlation = cov(x, y) / float(sd(x) * sd(y))
    return correlation


print('Correlation of X and Y is ' + str(corr(x, y)))
