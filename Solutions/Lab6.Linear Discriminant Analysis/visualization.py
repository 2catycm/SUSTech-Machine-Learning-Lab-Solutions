from matplotlib import pyplot as plt
def draw_scatter(featureIndexA, featureIndexB, sklearn_bunch):
    """画图

    Args:
        featureIndexA (_type_): _description_
        featureIndexB (_type_): _description_
        sklearn_bunch (_type_): _description_
    """
    plt.figure()
    X, y = sklearn_bunch.data, sklearn_bunch.target
    for target in sklearn_bunch.target_names:
            plt.scatter(X[y==target, featureIndexA], 
                        X[y==target, featureIndexB])
