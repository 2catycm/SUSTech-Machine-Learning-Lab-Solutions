from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
def evaluate(pipeline, data, target, cross=True):
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    # pipeline.fit(X_train, y_train)
    if cross:
        scores = cross_val_score(pipeline, X_train, y_train) # 没有指定estimator，使用模型默认的estimator，就是R2。
        return scores.mean()
    else:
        return pipeline.fit(X_train, y_train).score(X_test, y_test)



from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# Evaluating Model
from sklearn.metrics import confusion_matrix, accuracy_score, fbeta_score, classification_report, roc_auc_score, roc_curve
def evaluate_multiclass(model, X_test, y_test, pos_class=0, algs_name=None):
    if algs_name is None:
        # algs_name = model.__name__
        algs_name = "model"
    y_pred = model.predict(X_test) 
    print(f'{algs_name} accuracy = {accuracy_score(y_pred,y_test)}')
    try:
        y_prob = model.predict_proba(X_test)[:,pos_class]
    except:
        y_prob = model.decision_function(X_test)
    y_test_new = y_test.copy()
    y_test_new[y_test==pos_class] = 1
    y_test_new[y_test!=pos_class] = 0
    fper, tper, thresholds = roc_curve(y_test_new, y_prob, pos_label=1)
    # print(f'{algs_name} f_score = {fbeta_score(y_pred,y_test, beta=1)}')
    plt.figure()
    plt.plot(fper, tper, color='orange', label='ROC')
    print(f'fper={fper}, tper={tper}, thresholds={thresholds}')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve for {algs_name}')
    plt.legend()
    plt.show()