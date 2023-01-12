import numpy as np


class OGH:

    """
    Implementation of an Optimized Geometric Hermite curve
    Taking in two endpoints and their endpoint vectors, it produces the curve with the minimum strain energy
    https://www.cs.uky.edu/~cheng/PUBL/Paper-Geometric-Hermite.pdf
    """


    def __init__(self, pt_0, vec_0, pt_1, vec_1):
        self.pt_0 = np.array(pt_0)
        self.vec_0 = np.array(vec_0)
        self.pt_1 = np.array(pt_1)
        self.vec_1 = np.array(vec_1)
        self.len_approx = None

    @property
    def dim(self):
        return len(self.pt_0)

    @property
    def a0(self):
        diff = self.pt_1 - self.pt_0
        numer = 6 * diff.dot(self.vec_0) * self.vec_1.dot(self.vec_1) \
              - 3 * diff.dot(self.vec_1) * self.vec_0.dot(self.vec_1)
        denom = 4 * self.vec_0.dot(self.vec_0) * self.vec_1.dot(self.vec_1) \
              - (self.vec_0.dot(self.vec_1)) ** 2

        return numer / denom

    @property
    def a1(self):
        diff = self.pt_1 - self.pt_0
        numer = 3 * diff.dot(self.vec_0) * self.vec_0.dot(self.vec_1) \
              - 6 * diff.dot(self.vec_1) * self.vec_0.dot(self.vec_0)
        denom = (self.vec_0.dot(self.vec_1)) ** 2 \
              - 4 * self.vec_0.dot(self.vec_0) * self.vec_1.dot(self.vec_1)

        return numer / denom

    @staticmethod
    def scalar_array_mul(scalars, vec):
        return scalars.reshape(-1, 1) * vec.reshape(1, -1)

    def eval(self, samples=101):
        ts = np.linspace(0, 1, samples, endpoint=True)
        pts = self.scalar_array_mul((2 * ts + 1) * (ts - 1) ** 2, self.pt_0) \
            + self.scalar_array_mul((-2 * ts + 3) * ts ** 2, self.pt_1) \
            + self.scalar_array_mul((1 - ts) ** 2 * ts * self.a0, self.vec_0) \
            + self.scalar_array_mul((ts - 1) * ts ** 2 * self.a1, self.vec_1)

        self.len_approx = np.sum(np.linalg.norm(pts[:-1] - pts[1:], axis=1))
        return pts

    @property
    def approx_len(self):
        if self.len_approx is None:
            self.eval()
        return self.len_approx

    @property
    def strain(self):
        A = 12 * self.pt_0 - 12 * self.pt_1 + 6 * self.a0 * self.vec_0 + 6 * self.a1 * self.vec_1
        B = -6 * self.pt_0 + 6 * self.pt_1 - 4 * self.a0 * self.vec_0 - 2 * self.a1 * self.vec_1
        return (A.dot(A) / 3 + A.dot(B) + B.dot(B)) / self.approx_len

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # curve = OGH([0, 0], [1, 1], [5, 0], [1, 0])
    # pts = curve.eval()
    #
    # plt.plot(pts[:,0], pts[:,1])
    # plt.axis('equal')
    # plt.show()
    #
    curve = OGH([0, 0, 0], [1, 0, 0], [3, 3, 3], [0, 1, 0])
    pts = curve.eval()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pts[:,0], pts[:,1], pts[:,2])
    plt.show()

    curve_1 = OGH([0, 0], [1, 1], [5, 0], [1, 0])
    curve_2 = OGH([0, 0], [1, 1], [10, 0], [1, 0])
    curve_3 = OGH([0, 0], [1, 1], [5, 0], [-1, -1])

    print(f'Curve 1:\n\tStrain: {curve_1.strain}\n\tLength: {curve_1.approx_len}')
    print(f'Curve 2:\n\tStrain: {curve_2.strain}\n\tLength: {curve_2.approx_len}')
    print(f'Curve 3:\n\tStrain: {curve_3.strain}\n\tLength: {curve_3.approx_len}')

    print()

