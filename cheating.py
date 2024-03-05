from sklearn.svm import LinearSVC, SVC
from data import load_data
from evaluate import compute_metrics

if __name__ =="__main__":
    x_train, y_train, x_test, y_test = load_data(test_size = 0.15)
    # classifier = LinearSVC(penalty='l2', loss='hinge')
    classifier = SVC(kernel='poly')
    classifier.fit(x_train, y_train)
    compute_metrics(classifier, x_test, y_test, 'cheating_svc_poly')