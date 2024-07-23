import taichi as ti
from sand_simulator import SandSimulator
from apic_extension import FluidSimulator_APIC
from initializer_2d import *


@ti.data_oriented
class CouplingSandWater:
    def __init__(self, quality=1,dt = 1e-4):
        # 计算设置
        self.quality = quality
        self.dt = dt
        self.g = ti.Vector([0, -10])

        # 耦合参数
        self.porosity, self.permeability = 0.4, 0.6                                # 孔隙度和渗透率
        self.b = ti.field(dtype=ti.f32, shape=2)                         # 右手项：外力与内力引起的动量
        self.X = ti.field(dtype=ti.f32, shape=2)                         # 解初始化为上一时间步速度——>更快收敛

        self.sand_sim = SandSimulator(100, dt = 1e-4)                    # todo 将网格尺寸和区域修改
        self.water_sim = FluidSimulator_APIC(res = (100, 100), dt = 1e-4)

        self.initializer_pos_s = Initializer2D(dim=2, particle_diameter=0.002)
        self.initializer_pos_w = Initializer2D(dim=2, particle_diameter=0.002)


    @ti.kernel
    def exchange_momentum(self):
        for k in ti.static(range(2)):  # todo

            for i, j in self.sand_sim.grid_m_s[k]:  # todo 共两个维度的网格容器

                if self.sand_sim.grid_m_s[k][i, j] > 0 and self.water_sim.grid_m[k][i, j] > 0:        # 是否耦合？

                    cE = (self.porosity * self.porosity * self.water_sim.rho * self.g[1]) / self.permeability  # 拖拽力系数
                    ms, mw = self.sand_sim.grid_m_s[k][i, j], self.water_sim.grid_m[k][i, j]
                    vs, vw = self.sand_sim.grid_v_s[k][i, j], self.water_sim.velocity[k][i, j]

                    d = cE * mw * ms * self.dt                                                  # 相互作用力系数
                    M = ti.Matrix([[ms, 0], [0, mw]])                                           # 质量矩阵
                    D = ti.Matrix([[-d, d], [d, -d]])                                           # todo 符号是否正确？

                    self.b[0] = ms * vs                                                         # todo 在与不可压缩流体耦合前，已经考虑了重力和内力的影响
                    self.b[1] = mw * vw

                    A = M + D                                                                   # 左手项系数
                    self.X[0], self.X[1] = vs, vw

                    for it in ti.static(range(3)):                                              # 高斯赛德尔迭代
                        for i in ti.static(range(2)):                                           # 两相
                            sum_Ax = 0.0
                            for j in ti.static(range(2)):                                       # 二维
                                if i != j: sum_Ax += A[i, j] * self.X[j]
                            self.X[i] = (self.b[i] - sum_Ax) / A[i, i]

                    self.sand_sim.grid_v_s[k][i, j], self.water_sim.velocity[k][i, j] = self.X[0], self.X[1]

    @ti.kernel
    def cal_saturation(self):
        for p in range(self.sand_sim.n_particle_s[None]):
            for k in ti.static(range(2)):                                                            # todo 分维度进行插值
                stagger = 0.5 * (1 - ti.Vector.unit(2, k))  # [0,0.5],[0.5,0]
                base = (self.sand_sim.x_s[p] / self.sand_sim.dx - (stagger + 0.5)).cast(ti.i32)
                fx = self.sand_sim.x_s[p] / self.sand_sim.dx - (base.cast(ti.f32) + stagger)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline

                self.sand_sim.saturation[p] = 0
                for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
                    weight = w[i][0] * w[j][1]
                    if self.sand_sim.grid_m_s[k][base + ti.Vector([i, j])] > 0 and self.water_sim.grid_m[k][base + ti.Vector([i, j])] > 0:
                        self.sand_sim.saturation[p] += weight   # 计算饱和度


    def initialize(self):
        self.initializer_pos_s.init_scene_sand(self.sand_sim)
        self.initializer_pos_w.init_scene_water(self.water_sim)

        self.sand_sim.initialize_s()

    def substep(self):
        self.water_sim.p2g()
        self.sand_sim.p2g_s()

        self.sand_sim.update_grid_s()
        self.sand_sim.apply_force_s()

        self.water_sim.identify_fluid_p()                                           # 根据粒子位置，划分 FLUID 和 AIR
        self.water_sim.enforce_boundary()                                           # 施加边界
        self.water_sim.add_gravity()                             # 在网格上施加重力加速度
        self.water_sim.solve_pressure(self.water_sim.strategy)   # 调用 mgpcg求解器
        self.water_sim.apply_pressure()                                             # 通过压力修改粒子速度

        self.exchange_momentum()                                                    # 调用耦合模块
        self.cal_saturation()

        self.sand_sim.enforce_boundary_s()

        self.sand_sim.g2p_s()
        self.water_sim.g2p()

        if self.water_sim.consider_tension:
            self.water_sim.surface.build_surface(self.water_sim)
            self.water_sim.add_tension()                                        # 表面粒子张力 ——> 流体网格张力
            self.water_sim.add_tension_to_particle()                            # 流体网格张力 ——> 粒子速度

        self.water_sim.advect_particles()
        self.water_sim.enforce_boundary()




