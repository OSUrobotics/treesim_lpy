import numpy as np
import scipy.optimize as opt

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

def normalize(vec):
    return vec / np.linalg.norm(vec)

def run_curve_strain_opt(all_pts, start_vector, spring_constant=1.0):
    # Attempts to find a curve that minimizes the energy of a curve that starts at the start point with the given
    # vector and passes through the target points.

    start_vector = normalize(start_vector)
    dim = len(start_vector)
    num_pts = len(all_pts)

    if dim == 2:
        def deparametrize(u):
            return np.array([np.cos(u), np.sin(u)])
        def parametrize(vec):
            vec = normalize(vec)
            return np.arctan2(vec[1], vec[0])
    elif dim == 3:
        def deparametrize(u, v):
            return np.array([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)])
        def parametrize(vec):
            vec = normalize(vec)
            return np.array([np.arccos(vec[2]), np.sign(vec[1]) * np.arccos(vec[0] / np.linalg.norm(vec[:2]))])
    else:
        raise NotImplementedError()

    def opt_func(params):
        # Vec is len (num_pts + 1) * (dim - 1)
        total_energy = 0
        all_vecs = [deparametrize(*params[i * (dim - 1):(i+1) * (dim-1)]) for i in range(num_pts)]

        for i in range(len(all_vecs) - 1):
            curve = OGH(all_pts[i], all_vecs[i], all_pts[i + 1], all_vecs[i + 1])
            total_energy += curve.strain

        start_angle_diff = np.arccos(start_vector.dot(all_vecs[0]))
        total_energy += 0.5 * spring_constant * start_angle_diff ** 2

        return total_energy

    init_guesses = []
    for i, target_pt in enumerate(all_pts):
        if i == 0:
            init_guesses.append(start_vector)
            continue
        last_pt = all_pts[i-1]
        next_pt = all_pts[i+1] if i < len(all_pts) - 1 else target_pt
        init_guesses.append(normalize(next_pt - last_pt))
    start_guess = np.array([parametrize(guess) for guess in init_guesses]).flatten()
    print('Start val: {:.4f}'.format(opt_func(start_guess)))

    rez = opt.minimize(opt_func, start_guess, bounds=[(-np.pi, np.pi)] * len(start_guess))
    final_vecs = [deparametrize(*rez.x[i * (dim - 1):(i+1) * (dim-1)]) for i in range(num_pts)]

    print('Final val: {:.4f}'.format(opt_func(rez.x)))

    return final_vecs, init_guesses



if __name__ == '__main__':

    pts = np.array([[0,0], [5,1], [2,5]])
    start_vec = normalize(np.array([1,2]))
    k = 1.0

    final_vecs, init_guesses = run_curve_strain_opt(pts, start_vec, k)

    import matplotlib.pyplot as plt

    for i in range(len(pts) - 1):
        curve = OGH(pts[i], init_guesses[i], pts[i+1], init_guesses[i+1])
        # curve = OGH(pts[i], final_vecs[i], pts[i+1], final_vecs[i+1])
        curve_pts = curve.eval()
        plt.plot(curve_pts[:,0], curve_pts[:,1])

    plt.scatter(pts[:,0], pts[:,1], color='red')
    plt.arrow(x=pts[0][0], y=pts[0][1], dx=start_vec[0], dy=start_vec[1])

    for pt, vec, init_guess in zip(pts, final_vecs, init_guesses):
        plt.arrow(x=pt[0], y=pt[1], dx=vec[0], dy=vec[1], color='blue', linestyle='solid')
        plt.arrow(x=pt[0], y=pt[1], dx=init_guess[0], dy=init_guess[1], color='red', linestyle='dashed')
    plt.axis('equal')
    plt.show()
