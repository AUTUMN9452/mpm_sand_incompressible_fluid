from math import pi
import taichi as ti
import numpy as np
from visualize_2d import *
from initializer_2d import *

@ti.data_oriented
class SandSimulator:
    def __init__(self, n_grid=100, dt = 1e-4):
        # 计算设置
        self.length = 1
        self.n_grid = n_grid  # 网格数
        self.dx, self.inv_dx = self.length / self.n_grid, float(self.n_grid)
        self.dt = dt  # 时间增量步
        self.g = ti.Vector([0, -10])  # 重力
        self.dim = 2
        self.max_num_s = 50000

        # 砂土材质
        self.n_particle_s = ti.field(dtype=ti.i32, shape=())                             # 真实粒子数
        self.n_particle_s[None] = 0
        # 砂土材质
        self.rho_s = 1500  # 粒子密度
        self.volume0_s = (self.dx * 0.5) ** 2  # 粒子体积
        self.m_s = self.rho_s * self.volume0_s  # 粒子质量

        self.E, self.nu = 10e6, 0.3  # 刚度、泊松比
        self.G, self.K = int(self.E / (2 * (1 + self.nu))), int(self.E / (3 * (1 - 2 * self.nu)))  # 剪切刚度、体积模量
        self.lambda_0 = int(self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)))  # 计算拉梅常数，剪切模量
        self.f_a, self.d_a, self.cohe = 30 * pi / 180, 10 * pi / 180, 5000  # 摩擦角、膨胀角和粘聚力
        self.q_f, self.q_d = 3 * ti.tan(self.f_a) / ti.sqrt(9 + 12 * ti.tan(self.f_a) ** 2), \
                             3 * ti.tan(self.d_a) / ti.sqrt(9 + 12 * ti.tan(self.d_a) ** 2)
        self.k_f0 = 3 * self.cohe / ti.sqrt(9 + 12 * ti.tan(self.f_a) ** 2)
        self.a_B = ti.sqrt(1 + self.q_f ** 2) - self.q_f
        self.max_t_s = self.cohe / ti.tan(self.f_a)  # 最大拉伸强度 maximum tensile strength
        self.H = 1000    # 塑性模量

        # 数据容器——质量点
        self.es_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 偏应变
        self.ev_s = ti.field(dtype=float, shape=self.max_num_s)  # 体积应变
        self.delta_e_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 应变增量
        self.delta_es_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 偏应变增量
        self.delta_ev_s = ti.field(dtype=float, shape=self.max_num_s)  # 体积应变增量
        self.omiga_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 自旋张量
        self.sigma_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 应力
        self.S_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 偏应力
        self.S_m = ti.field(dtype=float, shape=self.max_num_s)  # 球应力——标量
        self.volume_s = ti.field(dtype=float, shape=self.max_num_s)  # 粒子体积
        self.x_s = ti.Vector.field(2, dtype=float, shape=self.max_num_s)  # 粒子坐标位置
        self.v_s = ti.Vector.field(2, dtype=float, shape=self.max_num_s)  # 粒子速度
        # self.C_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 仿射速度场
        self.C_s = [ti.Vector.field(2, dtype=ti.f32, shape=self.max_num_s) for _ in range(2)]  # 仿射速度

        self.F_s = ti.Matrix.field(2, 2, dtype=float, shape=self.max_num_s)  # 形变梯度
        # hardening
        self.k_f = ti.field(dtype=float, shape=self.max_num_s)  # 强度变量，反映硬化
        # softening
        self.saturation = ti.field(dtype=float, shape=self.max_num_s)  # 饱和度

        # 数据容器——网格
        # self.grid_v_s = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))  # 网格点动量或速度
        # self.grid_m_s = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))  # 网格点质量
        # self.grid_f_s = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))  # 网格点力

        self.grid_v_s = [ti.field(dtype=ti.f32, shape=([self.n_grid + (d == i) for i in range(2)])) for d in
                         range(2)]  # [u = [n_grid+1, n_grid]; v = [n_grid, n_grid+1]]
        self.grid_m_s = [ti.field(dtype=ti.f32, shape=([self.n_grid + (d == i) for i in range(2)])) for d in
                         range(2)]
        self.grid_f_s = [ti.field(dtype=ti.f32, shape=([self.n_grid + (d == i) for i in range(2)])) for d in
                         range(2)]


    @ti.func
    def project(self, p):  # 塑性投影——将试探应力拉回至屈服面
        St_s = self.S_s[p] + (
                    self.S_s[p] @ self.omiga_s[p].transpose() + self.omiga_s[p] @ self.S_s[p]) * self.dt + 2 * self.G * \
               self.delta_es_s[p]  # 试探偏应力
        St_m = self.S_m[p] + self.K * self.delta_ev_s[p]  # 试探球应力
        St_t = ti.sqrt(0.5 * (St_s[0, 0] ** 2 + St_s[1, 1] ** 2 + 2 * St_s[1, 0] ** 2))  # 等效剪切应力
        delta_lam = 0.0  # 塑性变形大小

        self.k_f[p] = self.k_f[p] * (1 - self.saturation[p])
        max_t_s = self.k_f[p] / self.q_f

        fs = St_t + self.q_f * St_m - self.k_f[p]  # 剪切屈服方程
        hs = St_t - self.a_B * (St_m - max_t_s)  # 拉伸屈服方程
        if St_m < max_t_s:
            if fs > 0:  # 剪切破坏
                delta_lam = fs / (self.G + self.K * self.q_f * self.q_d + self.H)
                self.k_f[p] = self.k_f[p] + self.H * delta_lam
                self.S_m[p] = St_m - self.K * delta_lam * self.q_d  # 更新球应力
                S_t = self.k_f[p] - self.q_f * self.S_m[p]  # 更新剪切应力
                self.S_s[p] = S_t / St_t * St_s  # 更新偏应力
                self.sigma_s[p] = self.S_s[p] + self.S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）
            else:  # 未发生破坏
                self.S_m[p] = St_m  # 更新球应力
                self.S_s[p] = St_s  # 更新偏应力
                self.sigma_s[p] = self.S_s[p] + self.S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）
        elif St_m >= max_t_s:
            if hs > 0:  # 剪切破坏
                delta_lam = fs / (self.G + self.K * self.q_f + self.H)
                self.k_f[p] = self.k_f[p] + self.H * delta_lam
                self.S_m[p] = St_m - self.K * delta_lam * self.q_d  # 更新球应力
                S_t = self.k_f[p] - self.q_f * self.S_m[p]  # 更新剪切应力
                self.S_s[p] = S_t / St_t * St_s  # 更新偏应力
                self.sigma_s[p] = self.S_s[p] + self.S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）
            else:  # 拉伸破坏
                self.S_m[p] = max_t_s  # 更新球应力
                self.S_s[p] = St_s  # 更新偏应力
                self.sigma_s[p] = self.S_s[p] + self.S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）


    @ti.kernel
    def p2g_s(self):  # 根据粒子信息更新网格节点的力、质量、动量
        for k in ti.static(range(self.dim)):  # 网格速度 重置为 0
            self.grid_v_s[k].fill(0)
            self.grid_f_s[k].fill(0)
            self.grid_m_s[k].fill(0)

        for p in range(self.n_particle_s[None]):  # 历遍所有粒子
            p_c = ti.Matrix([[self.C_s[0][p][0], self.C_s[0][p][1]], [self.C_s[1][p][0], self.C_s[1][p][1]]])   # todo 仿射矩阵合并
            self.F_s[p] = (ti.Matrix.identity(float, 2) + self.dt * p_c) @ self.F_s[p]  # 形变梯度矩阵
            e_dot = 0.5 * (p_c + p_c.transpose())  # 应变率
            self.delta_e_s[p] = e_dot * self.dt  # 应变增量
            self.omiga_s[p] = 0.5 * (p_c - p_c.transpose())  # 旋转速率
            self.delta_ev_s[p] = self.delta_e_s[p].trace()  # 体积应变增量
            self.volume_s[p] = self.volume_s[p] / (1 + self.delta_ev_s[p])  # 体积变化

            self.delta_es_s[p] = self.delta_e_s[p] - 0.5 * self.delta_ev_s[p] * ti.Matrix.identity(float, 2)  # 偏应变增量
            self.project(p)
            for k in ti.static(range(self.dim)):                                                            # todo 分维度进行插值
                stagger = 0.5 * (1 - ti.Vector.unit(self.dim, k))  # [0,0.5],[0.5,0]
                base = (self.x_s[p] / self.dx - (stagger + 0.5)).cast(ti.i32)
                fx = self.x_s[p] / self.dx - (base.cast(ti.f32) + stagger)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline
                w_grad = [fx - 1.5, 2 - 2 * fx, fx - 0.5]  # 权重梯度

                for i, j in ti.static(ti.ndrange(3, 3)):  # 对粒子周围3x3网格开始历遍
                    offset = ti.Vector([i, j])
                    dpos = (offset.cast(ti.f32) - fx) * self.dx
                    weight = w[i][0] * w[j][1]
                    weight_grad = ti.Matrix([w_grad[i][0] * w[j][1]/self.dx, w[i][0] * w_grad[j][1]/self.dx])

                    self.grid_v_s[k][base + offset] += weight * self.m_s * (self.v_s[p][k] + self.C_s[k][p].dot(dpos))  # 仿射速度
                    self.grid_m_s[k][base + offset] += weight * self.m_s  # 权重累加
                    force = self.volume_s[p] * self.sigma_s[p] @ weight_grad
                    self.grid_f_s[k][base + offset] += -force[k]  # todo MAC网格力如何更新

    @ti.kernel
    def update_grid_s(self):  # 动量——>速度
        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.grid_v_s[k]):  # 此时velocity_backup为权重和
                if self.grid_m_s[k][I] > 0:
                    self.grid_v_s[k][I] = self.grid_v_s[k][I]/self.grid_m_s[k][I]


    @ti.kernel
    def apply_force_s(self):  # 根据节点力更新速度
        for k in ti.static(range(self.dim)):
                gravity = self.g[k]
                for I in ti.grouped(self.grid_v_s[k]):
                    if self.grid_m_s[k][I] > 0:
                        self.grid_v_s[k][I] += self.grid_f_s[k][I] * self.dt / self.grid_m_s[k][I]  # 更新节点速度
                        self.grid_v_s[k][I] += gravity * self.dt

    @ti.kernel
    def enforce_boundary_s(self):
        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.grid_v_s[k]):
                if (I[k] <= 3) and self.grid_v_s[k][I] < 0:
                    self.grid_v_s[1][I] = 0
                    self.grid_v_s[0][I] = 0

                if (I[k] >= self.n_grid - 3) and self.grid_v_s[k][I] > 0:
                    self.grid_v_s[1][I] = 0
                    self.grid_v_s[0][I] = 0

    @ti.kernel
    def g2p_s(self):  # 通过节点更新粒子速度、仿射速度、位置
        for p in range(self.n_particle_s[None]):  # 历遍所有粒子
            for k in ti.static(range(self.dim)):                                                            # todo 分维度进行插值
                stagger = 0.5 * (1 - ti.Vector.unit(self.dim, k))  # [0,0.5],[0.5,0]
                base = (self.x_s[p] / self.dx - (stagger + 0.5)).cast(ti.i32)
                fx = self.x_s[p] / self.dx - (base.cast(ti.f32) + stagger)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline
                w_grad = [fx - 1.5, 2 - 2 * fx, fx - 0.5]  # 权重梯度

                new_c = ti.Vector([0.0, 0.0])
                new_v = 0.00  # 单个粒子新的速度容器
                for i, j in ti.static(ti.ndrange(3, 3)):
                    g_v = self.grid_v_s[k][base + ti.Vector([i, j])]  # 3x3中每个网格点的速度
                    weight = w[i][0] * w[j][1]
                    weight_grad = ti.Vector([w_grad[i][0] * w[j][1]/self.dx, w[i][0] * w_grad[j][1]/self.dx])
                    new_v += weight * g_v  # 重新根据3x3网格的 权重 计算粒子速度
                    new_c += weight_grad * g_v
                self.v_s[p][k] = new_v  # 粒子速度
                self.C_s[k][p] = new_c  # 仿射速度

        for p in self.x_s:  # 历遍所有粒子
            self.x_s[p] += self.dt * self.v_s[p]  # 根据速度计算位移




    @ti.kernel
    def initialize_s(self):
        for i in range(self.n_particle_s[None]):
            # self.x_s[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.3 + 0.03]
            self.v_s[i] = ti.Matrix([0, 0])
            self.F_s[i] = ti.Matrix([[1, 0], [0, 1]])
            self.volume_s[i] = self.volume0_s
            self.saturation[i] = 0

            self.sigma_s[i] = ti.Matrix([[0, 0], [0, 0]])                               # 柯西应力
            self.S_m[i] = (self.sigma_s[i][0, 0] + self.sigma_s[i][1, 1]) * 0.5         # 球应力
            self.S_s[i] = self.sigma_s[i] - self.S_m[i] * ti.Matrix.identity(float, 2)  # 偏应力
            self.k_f[i] = self.k_f0


    def sustep_s(self):
        self.p2g_s()
        self.update_grid_s()
        self.apply_force_s()

        self.enforce_boundary_s()

        self.g2p_s()



if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    color_array = ti.Vector.field(3, dtype=ti.f32, shape=2)
    color_array[0] = [0.2, 0.5, 0.8]  #
    color_array[1] = [1.0, 0.0, 0.0]  #

    sand_sim = SandSimulator()
    gui = ti.GUI("Taichi mac sand DP-model", res=512, background_color=0xFFFFFF)

    initializer_pos = Initializer2D(dim = 2, particle_diameter = 0.002)
    initializer_pos.init_scene_sand(sand_sim)
    sand_sim.initialize_s()

    visualizor = DataColorMapper(sand_sim.n_particle_s[None], color_array, res = (sand_sim.n_grid, sand_sim.n_grid))

    frame = 0
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(50):
            sand_sim.sustep_s()
        frame += 1

        min_value = min(sand_sim.k_f.to_numpy()[:sand_sim.n_particle_s[None]])
        visualizor.update_color_field(sand_sim.k_f, visual_range=(min_value, min_value+10))  #

        gui.circles(sand_sim.x_s.to_numpy()[:sand_sim.n_particle_s[None]], radius=1.5, color=visualizor.color.to_numpy())
        gui.show()