import numpy as np
import scipy.optimize as opt



class CubicBezier:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    @staticmethod
    def scalar_array_mul(scalars, vec):
        return scalars.reshape(-1, 1) * vec.reshape(1, -1)

    def __call__(self, t):
        return self.eval(t)

    def eval(self, t):
        return sum(self.scalar_array_mul(ts, pt) for ts, pt in [
            ((1-t) ** 3, self.p0),
            (3*(1-t) ** 2 * t, self.p1),
            (3*(1-t)*t**2, self.p2),
            (t**3, self.p3)
        ])

    def deriv(self, t):
        return sum(self.scalar_array_mul(ts, pt) for ts, pt in [
            (3*(1-t) ** 2, self.p1 - self.p0),
            (6*(1-t) * t, self.p2 - self.p1),
            (3*t**2, self.p3 - self.p2)
        ])

    def second_deriv(self, t):
        return sum(self.scalar_array_mul(ts, pt) for ts, pt in [
            (6*(1-t), self.p2 - 2 * self.p1 + self.p0),
            (6*t, self.p3 - 2 * self.p2 + self.p1)
        ])



class HermiteCurve:

    """
    Implementation of an Optimized Geometric Hermite curve
    Taking in two endpoints and their endpoint vectors, it produces the curve with the minimum strain energy
    https://www.cs.uky.edu/~cheng/PUBL/Paper-Geometric-Hermite.pdf
    """


    def __init__(self, pt_0, vec_0, a0, pt_1, vec_1, a1):
        self.pt_0 = np.array(pt_0)
        self.vec_0 = np.array(vec_0)
        self.a0 = a0
        self.pt_1 = np.array(pt_1)
        self.vec_1 = np.array(vec_1)
        self.a1 = a1
        self.len_approx = None

    @property
    def dim(self):
        return len(self.pt_0)

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


class OGH(HermiteCurve):
    def __init__(self, pt_0, vec_0, pt_1, vec_1):
        diff = pt_1 - pt_0
        numer = 6 * diff.dot(vec_0) * vec_1.dot(vec_1) \
                - 3 * diff.dot(vec_1) * vec_0.dot(vec_1)
        denom = 4 * vec_0.dot(vec_0) * vec_1.dot(vec_1) \
                - (vec_0.dot(vec_1)) ** 2

        a0 = numer / denom

        numer = 3 * diff.dot(vec_0) * vec_0.dot(vec_1) \
                - 6 * diff.dot(vec_1) * vec_0.dot(vec_0)
        denom = (vec_0.dot(vec_1)) ** 2 \
                - 4 * vec_0.dot(vec_0) * vec_1.dot(vec_1)

        a1 = numer / denom
        super().__init__(pt_0, vec_0, a0, pt_1, vec_1, a1)


def normalize(vec):
    return vec / np.linalg.norm(vec)

def unpack_cubic_bezier_opt_params(params, all_pts):
    dim = len(all_pts[0])
    num_pts = len(all_pts)
    control_points = params.reshape(-1, dim)
    curve_params = []
    for i in range(num_pts - 1):
        last_pt = all_pts[i]
        next_pt = all_pts[i + 1]
        lead_control = control_points[i]
        next_control = 2 * all_pts[i + 1] - control_points[i + 1] if i < num_pts - 2 else all_pts[i + 1]
        curve_params.append((last_pt, lead_control, next_control, next_pt))

    return curve_params

def run_cubic_bezier_strain_opt(all_pts, start_vector, spring_constant=1.0, curve_eval=201, max_opt=None, start_guess=None):

    start_vector = normalize(start_vector)
    dim = len(start_vector)
    num_pts = len(all_pts)

    def opt_func(vec):
        control_points = vec.reshape(-1, dim)
        total_energy = 0
        for curve_params in unpack_cubic_bezier_opt_params(vec, all_pts):
            curve = CubicBezier(*curve_params)
            ts = np.linspace(0, 1, curve_eval)
            pts = curve(ts)
            deriv = curve.deriv(ts)
            second_deriv = curve.second_deriv(ts)

            cross_product = np.abs(np.cross(deriv, second_deriv))
            if dim == 3:
                cross_product = np.linalg.norm(cross_product, axis=1)
            kappa = np.nan_to_num(cross_product / (np.linalg.norm(deriv, axis=1) ** 3))
            energy = np.sum(np.linalg.norm(pts[:-1] - pts[1:], axis=1) * kappa[1:] ** 2)
            total_energy += energy

        start_control_vec = normalize(control_points[0] - all_pts[0])
        angle_diff = np.arccos(np.clip(start_vector.dot(start_control_vec), -1, 1))
        total_energy += 0.5 * spring_constant * angle_diff ** 2

        return total_energy

    # Assemble the start guess
    if start_guess is None:
        start_guesses = []
        for i in range(num_pts - 1):

            pt = all_pts[i]
            next_pt = all_pts[i + 1]
            dist = np.linalg.norm(next_pt - pt)

            if i == 0:
                vec = normalize(start_vector)
            else:
                last_pt = all_pts[i - 1]
                vec = normalize(next_pt - last_pt)
            start_guesses.append(vec * dist / 3)
        start_guess = np.array(start_guesses).flatten()

    print('Start val: {:.3f}'.format(opt_func(start_guess)))
    options = {}
    if max_opt is not None:
        options['maxiter'] = max_opt
    rez = opt.minimize(opt_func, start_guess, method='Nelder-Mead', options=options)
    if not rez.success:
        print('Warning! Optimization did not succeed.\nMessage: {}'.format(rez.message))
    print('Final val: {:.3f}'.format(opt_func(rez.x)))

    return unpack_cubic_bezier_opt_params(rez.x, all_pts), unpack_cubic_bezier_opt_params(start_guess, all_pts), rez



# TESTING FUNCTIONS
def curve_opt_test_2d():
    # targets = np.array([[0,0], [5,2], [2.5,5]])
    targets = np.array([[0, 0], [3, 2], [6, 5]])
    start_vec = normalize(np.array([3, 1]))
    k = 10.0
    init_guess = run_cubic_bezier_strain_opt(targets, start_vec, 1)[2].x

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatch
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    fig, ax = plt.subplots()
    num_curves = len(targets) - 1
    plots = []
    arrows = []
    for _ in range(num_curves):
        plot, = ax.plot([], [], color='red')
        plots.append(plot)
        for _ in range(2):
            arrow = mpatch.FancyArrowPatch((-5, -5), (-10, -10))
            ax.add_patch(arrow)
            arrows.append(arrow)

    def init():
        pt_mins = targets.min(axis=0)
        pt_maxs = targets.max(axis=0)
        ax.set_xlim(pt_mins[0] - 2, pt_maxs[0] + 2)
        ax.set_ylim(pt_mins[1] - 2, pt_maxs[1] + 2)
        ax.add_patch(
            mpatch.FancyArrowPatch(targets[0], targets[0] + normalize(start_vec) * 2, color='red', arrowstyle='->'))

        ax.scatter(targets[:, 0], targets[:, 1], marker='*')

    def update(frame):
        k = frame / 2
        opt_curves, _, opt = run_cubic_bezier_strain_opt(targets, start_vec, k, start_guess=init_guess)
        ts = np.linspace(0, 1, 101)

        all_ctrls = []

        for curve_params, plot in zip(opt_curves, plots):
            curve = CubicBezier(*curve_params)
            pts = curve(ts)
            plot.set_data(pts[:, 0], pts[:, 1])
            ax.set_title('k={:.1f}, Energy={:.2f}'.format(k, opt.fun))
            all_ctrls.extend([(curve.p0, curve.p1), (curve.p3, curve.p2)])

        for arrow, ctrl in zip(arrows, all_ctrls):
            arrow.set_positions(*ctrl)

    ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init)
    ani.save('curve_opt_test.gif', PillowWriter(fps=5))


def curve_opt_test_3d():
    # targets = np.array([[0, 0, 0], [3, 1.5, 2], [1.5, -1.5, 5]])
    targets = np.array([[0, 0, 0], [3, 0.5, 2], [6, 0.25, 5]])
    start_vec = normalize(np.array([3, -0.1, 1]))
    init_guess = run_cubic_bezier_strain_opt(targets, start_vec, 1)[2].x

    from mpl_toolkits.mplot3d import axes3d, proj3d
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatch
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    class Arrow3D(mpatch.FancyArrowPatch):

        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs)

        def set_positions_3d(self, x, y, z, dx, dy, dz):
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    num_curves = len(targets) - 1
    plots = []
    projection_plots = []
    projection_plots_xz = []
    arrows = []
    for _ in range(num_curves):
        plot, = ax.plot([], [], [], color='red')
        plots.append(plot)

        proj_plot, = ax.plot([], [], [], color='grey', linestyle='dashed')
        projection_plots.append(proj_plot)

        proj_plot_xz, = ax.plot([], [], [], color='grey', linestyle='dashed')
        projection_plots_xz.append(proj_plot_xz)

        for _ in range(2):
            arrow = Arrow3D(*(-5, -5, -10), *(-10, -10, -10))
            ax.add_patch(arrow)
            arrows.append(arrow)

    def init():
        pt_mins = targets.min(axis=0)
        pt_maxs = targets.max(axis=0)
        ax.set_xlim(pt_mins[0] - 0.5, pt_maxs[0] + 0.5)
        ax.set_ylim(pt_mins[1] - 0.5, pt_maxs[1] + 0.5)
        ax.set_zlim(pt_mins[2] - 0.5, pt_maxs[2] + 0.5)
        ax.add_patch(
            Arrow3D(*targets[0], *normalize(start_vec) * 2, color='red', arrowstyle='->'))

        ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], marker='*')
        ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2] * 0, marker='*', color='grey')

    def update(frame):
        k = frame / 2
        opt_curves, _, opt = run_cubic_bezier_strain_opt(targets, start_vec, k, start_guess=init_guess)
        ts = np.linspace(0, 1, 101)

        all_ctrls = []

        for curve_params, plot, proj_plot, proj_plot_xz, in zip(opt_curves, plots, projection_plots, projection_plots_xz):
            curve = CubicBezier(*curve_params)
            pts = curve(ts)
            plot.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            proj_plot.set_data_3d(pts[:, 0], pts[:,1], np.zeros(pts.shape[0]))
            proj_plot_xz.set_data_3d(pts[:, 0], np.ones(pts.shape[0]) * ax.get_ylim()[1] , pts[:,2])
            ax.set_title('k={:.1f}, Energy={:.2f}'.format(k, opt.fun))
            all_ctrls.extend([(curve.p0, curve.p1), (curve.p3, curve.p2)])

        for arrow, ctrl in zip(arrows, all_ctrls):
            start, end = ctrl
            arrow.set_positions_3d(*start, *end - start)

    ani = FuncAnimation(fig, update, frames=np.arange(30), init_func=init)
    ani.save('curve_opt_test.gif', PillowWriter(fps=5))





if __name__ == '__main__':
    curve_opt_test_3d()


    #
    #
    #
    #
    # opt_curves, start_curves = run_cubic_bezier_strain_opt(targets, start_vec, k, max_opt=10000)
    # ts = np.linspace(0, 1, 101)
    # for curve_params in opt_curves:
    #     curve = CubicBezier(*curve_params)
    #     pts = curve(ts)
    #     plt.plot(pts[:,0], pts[:,1])
    #     plt.arrow(x=curve.p0[0], y=curve.p0[1], dx=(curve.p1 - curve.p0)[0], dy=(curve.p1 - curve.p0)[1], linestyle='dashed')
    #     plt.arrow(x=curve.p3[0], y=curve.p3[1], dx=(curve.p2 - curve.p3)[0], dy=(curve.p2 - curve.p3)[1], linestyle='dashed')
    #
    # plt.scatter(targets[:,0], targets[:,1], color='red')
    # plt.arrow(x=targets[0][0], y=targets[0][1], dx=start_vec[0], dy=start_vec[1], color='red')
    #
    # plt.axis('equal')
    # plt.show()
    #
    #





    # ctrl_points = [np.array(x) for x in [
    #     (0, 0), (1, 2), (2,-2), (3, 0)
    #     # (0, 0), (1, 3), (2, 3), (3, 0)
    #     # (0, 0), (0.1, 0.1), (0.9, 0.9), (1, 1)
    # ]]
    #
    # curve = CubicBezier(*ctrl_points)
    # ts = np.linspace(0, 1, 101)
    # pts = curve(ts)
    # deriv = curve.deriv(ts)
    # second_deriv = curve.second_deriv(ts)
    #
    # kappa = np.abs(np.cross(deriv, second_deriv)) / (np.linalg.norm(deriv, axis=1) ** 3)
    # normalized = (kappa - kappa.min()) / (kappa.max() - kappa.min())
    #
    # total = np.sum(np.linalg.norm(pts[:-1] - pts[1:], axis=1) * kappa[1:] ** 2)
    #
    # import matplotlib.pyplot as plt
    # import matplotlib.cm as cm
    #
    # colors = cm.rainbow(normalized)
    # plt.plot(pts[:,0], pts[:,1])
    # plt.scatter(pts[:,0], pts[:,1], color=colors)
    # plt.title('Total energy: {:.2f}'.format(total))
    # plt.show()