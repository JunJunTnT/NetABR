import numpy as np
from matplotlib import pyplot


class K_Means(object):
    def __init__(self, k, tolerance=0.0001, max_iter=300):
        self.clf_ = {}
        self.centers_ = {}
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            for j in range(self.k_):
                self.clf_[j] = []
            for feature in data:
                distances = []
                for center in self.centers_:
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index

    def clustered_by_type(self, features: list, device_info: list):
        classf = [[] for _ in range(self.k_)]
        for i in range(len(features)):
            for j in range(self.k_):
                for elem in self.clf_[j]:
                    if (features[i] == elem).all():
                        classf[j].append(i)
                        break
        default = False
        for elem in classf:
            if len(elem) == 0:
                default = True
                break
        if default:
            classf = [[] for _ in range(self.k_)]
            for i in range(self.k_):
                for j in range(len(device_info)):
                    if device_info[j] == i:
                        classf[i].append(j)

        return classf


if __name__ == '__main__':
    n = np.array([[1, 1, 1], [1, 1.1, 1], [1.5, 1.8, 5], [8, 8, 1], [8, 8, 9], [1, 0.6, 5]])
    x = [np.array([elem[0]]) for elem in n]
    k_means = K_Means(k=3)
    k_means.fit(x)
    print(k_means.centers_)
    print(k_means.clf_)

    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    pyplot.show()