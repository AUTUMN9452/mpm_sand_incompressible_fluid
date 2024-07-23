import taichi as ti
from functools import reduce
import time
from mgpcg import MGPCGPoissonSolver
from pressure_project import PressureProjectStrategy
from fluid_surface import Fluid_Surface
import interpolate_scheme as interpolate

from initializer_2d import *
from visualize_2d import *

# ti.init(arch=ti.cuda, kernel_profiler=False, device_memory_GB=4.0)
ti.init(arch=ti.gpu)


@ti.data_oriented
class FluidSimulator_APIC:
    def __init__(self,
                 dim=2,
                 res=(100, 100),
                 dt=1e-3,
                 rho=1000.0,
                 gravity=[0, -10],
                 p0=0.01):
        self.dim = dim
        self.length = 1

        self.res = res
        self.dx = self.length / self.res[0]
        self.inv_dx = 1/self.dx
        self.dt = dt
        self.total_t = 0.0                              # 总时间
        self.p0 = p0                                    # 标准大气压
        self.rho = rho                                  # 流体密度
        self.gravity = gravity                          # 重力
        self.volume0_w = (self.dx * 0.5) ** 2  # 粒子体积
        self.m_w = self.rho * self.volume0_w  # 粒子质量


        # parameter in cell
        self.cell_type = ti.field(dtype=ti.i32, shape = self.res)             # 网格类型
        self.pressure = ti.field(dtype=ti.f32, shape = self.res)              # 压力
        self.tension = ti.Vector.field(dim, dtype=ti.f32, shape = self.res)   # 表面张力
        self.velocity = [ti.field(dtype=ti.f32, shape=([res[i] + (d == i) for i in range(self.dim)])) for d in
                         range(self.dim)]                                     # (d=0时，容器为[res+1,res]; d=1时，容器为[res,res+1]])
        self.grid_m = [ti.field(dtype=ti.f32, shape=([res[i] + (d == i) for i in range(self.dim)])) for d in
                         range(self.dim)]

        # parameter in particle
        self.max_particles = reduce(lambda x, y : x * y, res) * (4 ** dim)        # 粒子的最大数量
        self.p_v = ti.Vector.field(dim, dtype=ti.f32, shape=self.max_particles)   # 粒子速度
        self.p_x = ti.Vector.field(dim, dtype=ti.f32, shape=self.max_particles)   # 粒子位置
        self.p_c = [ti.Vector.field(dim, dtype=ti.f32, shape=self.max_particles) for _ in range(self.dim)]  # 仿射速度
        self.p_num = ti.field(dtype=ti.i32, shape=())                             # 真实粒子数
        self.p_num[None] = 0
        self.p_material = ti.field(dtype=int, shape=self.max_particles)           # 粒子属性        todo 修改识别函数，将粒子材质考虑进来

        # parameter in fluid surface
        self.consider_tension = True
        self.tension_coefficient = 0.1


        # parameter in collision
        self.collision_factor = 0.5
        self.friction_coefficient = 0.5

        # extrap utils
        self.valid = ti.field(dtype=ti.i32, shape = ([res[_] + 1 for _ in range(self.dim)]))             # 用于标记 marker 是否有效
        self.valid_temp = ti.field(dtype=ti.i32, shape = ([res[_] + 1 for _ in range(self.dim)]))        # 临时存储 valid 字段的值

        # MGPCG
        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.iterations = 50
        self.verbose = True
        self.poisson_solver = MGPCGPoissonSolver(self.dim,
                                                 self.res,
                                                 self.n_mg_levels,
                                                 self.pre_and_post_smoothing,
                                                 self.bottom_smoothing)

        # Pressure Solve
        self.strategy = PressureProjectStrategy(self.dim,
                                                    self.velocity,
                                                    self.p0)
        # fluid surface
        self.surface = Fluid_Surface(self.dim, self.res[0], self.dx, self.consider_tension)


    @ti.func
    def is_valid(self, I):
        return all(I >= 0) and all(I < self.res)                            # 检测 I 的所有分量都大于0，或者小于res

    @ti.func
    def is_fluid(self, I):                                                  # 不为空 且 属性为 流体
        return self.is_valid(I) and self.cell_type[I] == 1

    @ti.func
    def is_solid(self, I):                                                  # 不为空 且 属性为 固体
        return not self.is_valid(I) or self.cell_type[I] == 2

    @ti.func
    def is_air(self, I):                                                    # 不为空 且 属性为 空气
        return self.is_valid(I) and self.cell_type[I] == 0


    @ti.kernel
    def p2g(self):  # 将粒子速度 p_v 映射至四周的网格上，通过线性插值的方法
        for k in ti.static(range(self.dim)):  # 网格速度 重置为 0
            self.velocity[k].fill(0)
            self.grid_m[k].fill(0)

        for p in range(self.p_num[None]):
            for k in ti.static(range(self.dim)):
                stagger = 0.5 * (1 - ti.Vector.unit(self.dim, k))  # [0,0.5],[0.5,0]
                interpolate.splat_vp_apic(self.dx, self.velocity[k], self.grid_m[k], self.p_x[p], self.p_v[p][k],  self.p_c[k][p], stagger, self.m_w)

        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.grid_m[k]):         # 此时grid_m为权重和
                if self.grid_m[k][I] > 0:                # 类似 动量/质量 的操作
                    self.velocity[k][I] /= self.grid_m[k][I]


    @ti.kernel
    def identify_fluid_p(self):                              # todo 根据粒子位置对流体区域进行识别
        for I in ti.grouped(self.cell_type):
            if not self.is_solid(I):
                self.cell_type[I] = 0
        for p in range(self.p_num[None]):
                pos = self.p_x[p]
                idx = ti.cast(ti.floor(pos / ti.Vector([self.dx, self.dx])), ti.i32)
                if not self.is_solid(idx):
                    self.cell_type[idx] = 1


    @ti.kernel
    def mark_valid(self, k : ti.template()):                           # 标记每个位置是否有效。有效位置包括流体和空气-液体界面
        for I in ti.grouped(self.velocity[k]):
            I_1 = I - ti.Vector.unit(self.dim, k)                      # [0,1],[1,0]
            if self.is_fluid(I_1) or self.is_fluid(I):                 # 前后只要有一个属于流体即有效
                self.valid[I] = 1
            else:
                self.valid[I] = 0

    @ti.kernel
    def enforce_boundary(self):                                                 # 设置固体周围的速度
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == 2:                                          # todo 左右侧边界横向速度为0，上下边界竖向速度为0
                if I[0] <= 2 or I[0] >= self.res[0] - 3:
                    self.velocity[0][I] = 0                                     # 将固体单元左边的速度 -> 0
                    self.velocity[0][I + ti.Vector.unit(self.dim, 0)] = 0       # 将固体单元右边[1,0]的速度也设置 -> 0
                elif I[1] <= 2 or I[1] >= self.res[0] - 3:
                    self.velocity[1][I] = 0                                     # 将固体单元左边的速度 -> 0
                    self.velocity[1][I + ti.Vector.unit(self.dim, 0)] = 0       # 将固体单元右边[1,0]的速度也设置 -> 0
                else:
                    self.velocity[0][I] = 0                                     # 将固体单元左边的速度 -> 0
                    self.velocity[0][I + ti.Vector.unit(self.dim, 0)] = 0       # 将固体单元右边[1,0]的速度也设置 -> 0
                    self.velocity[1][I] = 0                                     # 将固体单元左边的速度 -> 0
                    self.velocity[1][I + ti.Vector.unit(self.dim, 0)] = 0       # 将固体单元右边[1,0]的速度也设置 -> 0


    @ti.kernel
    def add_gravity(self):                                         # 施加重力加速度
        for k in ti.static(range(self.dim)):
            if ti.static(self.gravity[k] != 0):
                g = self.gravity[k]
                for I in ti.grouped(self.velocity[k]):
                    self.velocity[k][I] += g * self.dt


    def solve_pressure(self, strategy):                                     # 调用 mgpcg求解器
        strategy.scale_A = self.dt / (self.rho * self.dx * self.dx)
        strategy.scale_b = 1 / self.dx

        start1 = time.perf_counter()
        self.poisson_solver.reinitialize(self.cell_type, strategy)              # 构建泊松方程的系数矩阵 A 和源向量 b
        end1 = time.perf_counter()

        start2 = time.perf_counter()
        self.poisson_solver.solve(self.iterations, self.verbose)                # 根据多重网格 共轭梯度法 求解出压力
        end2 = time.perf_counter()

        print(f'\033[33minit cost {end1 - start1}s, solve cost {end2 - start2}s\033[0m')
        self.pressure.copy_from(self.poisson_solver.x)


    @ti.kernel
    def apply_pressure(self):          # 根据求解出来的压力更新速度
        scale = self.dt / (self.rho * self.dx)
        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.cell_type):
                I_1 = I - ti.Vector.unit(self.dim, k)                                           # 单元后边和左边
                if self.is_fluid(I_1) or self.is_fluid(I):                                      # 流体或者后边、左边是流体
                    if self.is_solid(I_1) or self.is_solid(I): self.velocity[k][I] = 0          # 有固体则直接为 0
                    # FLuid-Air
                    elif self.is_air(I):                                                        # 右边单元是空气
                        self.velocity[k][I] -= scale * (self.p0 - self.pressure[I_1])           # 根据压力更新 速度
                    # Air-Fluid
                    elif self.is_air(I_1):                                                      # 左边单元是空气
                        self.velocity[k][I] -= scale * (self.pressure[I] - self.p0)
                    # Fluid-Fluid
                    else: self.velocity[k][I] -= scale * (self.pressure[I] - self.pressure[I_1])    # 流体与流体

    @ti.kernel
    def g2p(self):
        for p in range(self.p_num[None]):
            for k in ti.static(range(self.dim)):
                stagger = 0.5 * (1 - ti.Vector.unit(self.dim, k))                                    # marker偏移值：[0,0.5],[0.5,0]
                self.p_v[p][k] = interpolate.sample_vp_apic(self.dx, self.velocity[k], self.p_x[p], stagger)  # APIC 粒子速度
                self.p_c[k][p] = interpolate.sample_cp_apic(self.dx, self.velocity[k], self.p_x[p], stagger)  # APIC 仿射速度矩阵


    @ti.kernel
    def advect_particles(self):                                                         # 应用摩擦与反弹
        for p in range(self.p_num[None]):
            pos = self.p_x[p]
            pv = self.p_v[p]
            # 施加粒子的边界，确保不会穿透固体
            collision_vec = ti.Vector.zero(ti.f32, self.dim)            # 碰撞
            for dim in ti.static(range(self.dim)):                      # dim的范围(0 1 2)，对应三个坐标值
                if pos[dim] >= (self.res[dim]-4) * self.dx:
                    collision_vec[dim] += 1.0                           # 用来判断碰撞后的方向
                    pos[dim] = (self.res[dim]-4) * self.dx              # 将位置约束在边界上
                elif pos[dim] <= (3) * self.dx:
                    collision_vec[dim] -= 1.0                           # 用来判断碰撞后的方向
                    pos[dim] = (3) * self.dx                            # 将位置约束在边界上
            collision_vec_normal = collision_vec.norm()
            if collision_vec_normal > 1e-6:
                vec = collision_vec / collision_vec.norm()              # 碰撞方向的单位矢量
                normal_component = pv.dot(vec) * vec                    # 速度在碰撞方向的分量
                tangent_component = pv - normal_component               # 速度在切向方向的分量
                pv -= self.friction_coefficient * tangent_component     # 添加摩擦
                pv -= (1.0 + self.collision_factor) * normal_component  # 反弹速度分量

            pos += pv * self.dt
            self.p_x[p] = pos                                           #
            self.p_v[p] = pv


    @ti.kernel
    def add_tension(self):                                                                  # 网格粒子 ——> 流体网格
        for I in ti.grouped(self.tension):                                                  # 初始化 张力容器
            self.tension[I] = ti.Vector([0.0, 0.0])
        for p in range(self.surface.surface_particle_num[None]):                            # 历遍所有表面粒子
            pos = self.surface.surface_particles[p].position
            base = int(pos* self.inv_dx - 0.5)                                              #
            fx = pos * self.inv_dx - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            normal, curvature = self.surface.linear_interpolation(self.surface.surface_particles.position[p])  # 线性加权周围网格的SDF梯度和曲率到表面粒子上
            # print("梯度是：", normal,"曲率是：", curvature)
            t = normal * curvature * self.tension_coefficient * self.dt                     # 根据 CSF张力公式计算
            for offset in ti.static(ti.grouped(ti.ndrange(*(3,) * self.dim))):
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.tension[base + offset] -= weight * t                                   # 表面粒子张力 ——> 流体网格张力


    # 将网格节点的表面张力映射给流体粒子。
    @ti.kernel
    def add_tension_to_particle(self):
        for p in range(self.p_num[None]):                                                # 流体网格张力 ——> MPM流体粒子速度
            base = (self.p_x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(ti.ndrange(*(3,) * self.dim))):
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.p_v[p] += weight * self.tension[base + offset]

    # --------------------------------------------------总分析步---------------------------------------------
    def initialize(self, initializer):
        initializer.init_scene(self)                            # 确定流体区域


    def substep(self):

        self.p2g()                                                  # 网格速度清零，粒子速度 -> 网格速度

        self.identify_fluid_p()                                     # 根据粒子位置，划分 FLUID 和 AIR
        self.enforce_boundary()                                     # 施加边界
        self.add_gravity()                                          # 在网格上施加重力加速度
        self.solve_pressure(self.strategy)                          # 调用 mgpcg求解器
        self.apply_pressure()                                       # 通过压力修改粒子速度

        self.g2p()                                                  # 将网格速度传回粒子

        if self.consider_tension:
            self.surface.build_surface(self)
            self.add_tension()                                        # 表面粒子张力 ——> 流体网格张力
            self.add_tension_to_particle()                            # 流体网格张力 ——> 粒子速度

        self.advect_particles()

        self.enforce_boundary()
        self.total_t += self.dt


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    initializer = Initializer2D(dim = 2, particle_diameter = 0.002)
    water_sim = FluidSimulator_APIC(res = (100, 100))
    initializer.init_scene_water(water_sim)

    gui = ti.GUI("Taichi water incompressible", res=512, background_color=0xFFFFFF)
    frame = 0

    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(50):  # 总时间帧
            water_sim.substep()
        frame += 1
        gui.circles(water_sim.p_x.to_numpy()[:water_sim.p_num[None]], radius=1.5, color=0x19A5DD)
        gui.show()