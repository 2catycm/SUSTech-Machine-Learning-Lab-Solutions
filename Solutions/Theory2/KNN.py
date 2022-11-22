import numpy as np
from sklearn import neighbors
def distanceFunc(metric_type, vec1, vec2):
    """
    Computes the distance between two d-dimension vectors. 
    Please DO NOT use Numpy's norm function when implementing this function. 
    Args:
        metric_type (str): Metric: L1, L2, or L-inf
        vec1 ((d,) np.ndarray): d-dim vector
        vec2 ((d,)) np.ndarray): d-dim vector
    Returns:
        distance (float): distance between the two vectors
    """

    diff = vec1 - vec2
    diff = np.abs(diff)
    distance = {"L1": np.sum(diff), "L2": np.sqrt(np.sum(diff ** 2)), "L-inf": np.max(diff)}[metric_type]
    return distance
def computeDistancesNeighbors(K, metric_type, X_train, y_train, sample):
    """
    Compute the distances between every datapoint in the train_data and the 
    given sample. Then, find the k-nearest neighbors.
    
    Return a numpy array of the label of the k-nearest neighbors.
    
    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        sample ((p,) np.ndarray): Single sample whose distance is to computed with every entry in the dataset
        
    Returns:
        neighbors (list): K-nearest neighbors' labels
    """
    distances = []
    for i in range(len(X_train)):
        distances.append(distanceFunc(metric_type, X_train[i], sample))
    distances = np.array(distances)
    argorder = np.argsort(distances, kind='stable')
    return y_train[argorder[:K]]
def Majority(neighbors):
    """
    Performs majority voting and returns the predicted value for the test sample.
    
    Since we're performing binary classification the possible values are [0,1].
    
    Args:
        neighbors (list): K-nearest neighbors' labels
        
    Returns:
        predicted_value (int): predicted label for the given sample
    """
    neighbors = neighbors.astype(int)
    return np.argmax(np.bincount(neighbors))

def KNN(K, metric_type, X_train, y_train, X_val):
    """
    Returns the predicted values for the entire validation or test set.
    
    Please DO NOT use Scikit's KNN model when implementing this function. 

    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        X_val ((n, p) np.ndarray): Validation or test data
        
    Returns:
        predicted_values (list): output for every entry in validation/test dataset 
    """
    predicted_values = []
    for i in range(len(X_val)):
        neighbors = computeDistancesNeighbors(K, metric_type, X_train, y_train, X_val[i])
        predicted_value = Majority(neighbors)
        predicted_values.append(predicted_value)
    return predicted_values

def main():
    """从标准输入读入，计算后从标准输出输出结果。
    input:
        - N(int): 训练集样本量
        - M(int): 验证集样本量
        - D(int): 特征的维度
        - x_train y_train (List<N, Tuple<List<D, float>, float>>): 训练集
        - x_test y_test (List<M, Tuple<List<D, float>, float>>): 验证集
    output:
        - K(int) in [0, 10] 多项式的阶数。选择使得测试集上标准差最小的K。
        - S(%.6f) 标准差 
    """
    N, M, D = map(int, input().split())
    x_train = []
    x_val = []
    y_train = []
    y_val = []
    for _ in range(N):
        line = input().split()
        x_train.append(list(map(float, line[:-1])))
        y_train.append(float(line[-1]))
    for _ in range(M):
        line = input().split()
        x_val.append(list(map(float, line[:-1])))
        y_val.append(float(line[-1]))
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    y_train = np.array(y_train, dtype='int')
    y_val = np.array(y_val, dtype='int')
    mi = min(y_train.min(), y_val.min())
    y_train-=mi
    y_val-=mi
    # print(x_train, y_train)
    # print(x_test, y_test)
    accs = {}    
    best_acc = 0
    for K in range(1, 6):
        for metric in ['L1', 'L2', 'L-inf']:
            y_pred = KNN(K, metric, x_train, y_train, x_val)
            y_pred = np.array(y_pred)
            # accuracy = (y_pred==y_val).mean()
            accuracy = int((y_pred==y_val).sum()) # 防止精度丢失
            if accuracy >= best_acc:
                best_acc = accuracy
                accs[accuracy] = accs.get(accuracy, [])+[(K, metric)]
    l = accs[best_acc]
    [print(f"{param[0]} {param[1]}") for param in l]
    # print(f"{best_acc}, l")
    # print(accs)
            
if __name__ == '__main__':
    main()