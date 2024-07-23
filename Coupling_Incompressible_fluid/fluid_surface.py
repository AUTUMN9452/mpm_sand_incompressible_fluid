import taichi as ti
ti.init(arch=ti.gpu)

@ti.data_oriented
class Fluid_Surface:
    def __init__(self, dim=2, diff_n_grid = 100, diff_dx = 1, consider_tension = True):
        self.dim = dim
        self.diff_n_grid = diff_n_grid              # 表面网格数
        self.diff_dx = diff_dx                      # 网格单元大小
        self.diff_inv_dx = 1 / self.diff_dx         # 倒数
        self.consider_tension = consider_tension    # 是否考虑张力

        # surface level set
        self.sign_distance_field = ti.field(ti.f32, shape=(self.diff_n_grid,) * dim)
        self.edge = ti.Struct.field({"begin_point": ti.types.vector(2, ti.f32),
                                     "end_point": ti.types.vector(2, ti.f32)}, shape=self.diff_n_grid ** 2)
        self.edge_num = ti.field(int, shape=())

        # surface tension
        self.SDF = ti.Struct.field({"gradient": ti.types.vector(2, ti.f32),
                                     "laplacian": float}, shape=(self.diff_n_grid,) * dim)      # SDF梯度和曲率
        self.surface_particle_num = ti.field(ti.i32, shape=())                                  # 表面粒子数
        self.surface_particles = ti.Struct.field({"position": ti.types.vector(dim, ti.f32)}, shape=(1000))


        self.radius = self.diff_dx * 0.8                                            # 粒子半径 = 网格间距
        self.mc_result = ti.field(ti.i32, shape=(self.diff_n_grid,) * dim)          # 液表编号

        self.trangle_table = ti.Vector.field(4, ti.i32, shape=(16))     # 液面表
        self.trangle_table[0] = ti.Vector([-1, -1, -1, -1])
        self.trangle_table[1] = ti.Vector([0, 1, -1, -1])
        self.trangle_table[2] = ti.Vector([0, 2, -1, -1])
        self.trangle_table[3] = ti.Vector([1, 2, -1, -1])
        self.trangle_table[4] = ti.Vector([1, 3, -1, -1])
        self.trangle_table[5] = ti.Vector([0, 3, -1, -1])
        self.trangle_table[6] = ti.Vector([1, 3, 0, 2])
        self.trangle_table[7] = ti.Vector([2, 3, -1, -1])
        self.trangle_table[8] = ti.Vector([2, 3, -1, -1])
        self.trangle_table[9] = ti.Vector([2, 3, 0, 1])
        self.trangle_table[10] = ti.Vector([0, 3, -1, -1])
        self.trangle_table[11] = ti.Vector([1, 3, -1, -1])
        self.trangle_table[12] = ti.Vector([1, 2, -1, -1])
        self.trangle_table[13] = ti.Vector([0, 2, -1, -1])
        self.trangle_table[14] = ti.Vector([0, 1, -1, -1])
        self.trangle_table[15] = ti.Vector([-1, -1, -1, -1])
# --------------------------------------------------表面识别---------------------------------------------
    # 生成level set隐式曲面
    @ti.kernel
    def gen_level_set(self, particle_pos : ti.template(), particle_num : ti.template()):
        for i, j in ti.ndrange(self.diff_n_grid, self.diff_n_grid):
            min_dis = 1000.0
            node_pos = ti.Vector([i * self.diff_dx, j * self.diff_dx])
            for I in range(particle_num):
                distance = (particle_pos[I] - node_pos).norm() - self.radius
                if distance < min_dis:
                    min_dis = distance  # 一个迭代过程，取该(i,j)网格到所有粒子的最小距离
            self.sign_distance_field[i, j] = min_dis

    @ti.func
    def gen_edge_pos(self, i, j, e):
        a = self.sign_distance_field[i, j]  # 获取当前网格单元的四个顶点的符号距离场值——>加权平均
        b = self.sign_distance_field[i + 1, j]
        c = self.sign_distance_field[i, j + 1]
        d = self.sign_distance_field[i + 1, j + 1]
        base_grid_pos = self.diff_dx * ti.Vector([i, j])  # 当前网格单元的基准位置
        result_pos = ti.Vector([.0, .0])
        if e == 0:  # 对应 ab边
            result_pos = base_grid_pos + ti.Vector([(abs(a) / (abs(a) + abs(b))) * self.diff_dx, 0])
        if e == 1:  # 对应 ac边
            result_pos = base_grid_pos + ti.Vector([0, (abs(a) / (abs(a) + abs(c))) * self.diff_dx])
        if e == 2:  # 对应 bd边
            result_pos = base_grid_pos + ti.Vector([self.diff_dx, (abs(b) / (abs(b) + abs(d))) * self.diff_dx])
        if e == 3:  # 对应 cd边
            result_pos = base_grid_pos + ti.Vector([(abs(c) / (abs(c) + abs(d))) * self.diff_dx, self.diff_dx])
        return result_pos  # 液体表面曲线的 首尾两端坐标

    # 将隐式曲面通过marching cube转化为显示曲面
    @ti.kernel
    def implicit_to_explicit(self):
        self.edge_num[None] = 0

        for i, j in ti.ndrange(self.diff_n_grid - 1, self.diff_n_grid - 1):
            self.mc_result[i, j] = 0  # 检查当前网格单元的四个顶点的符号距离场值，确定id = 0
            valueA = self.sign_distance_field[i, j]
            valueB = self.sign_distance_field[i + 1, j]
            valueC = self.sign_distance_field[i, j + 1]
            valueD = self.sign_distance_field[i + 1, j + 1]
            if valueA > 0.0:
                valueA = 1.0
            else:
                valueA = 0.0
            if valueB > 0.0:
                valueB = 1.0
            else:
                valueB = 0.0
            if valueC > 0.0:
                valueC = 1.0
            else:
                valueC = 0.0
            if valueD > 0.0:
                valueD = 1.0
            else:
                valueD = 0.0

            self.mc_result[i, j] = ti.cast(valueA * 1 + valueB * 2 + valueC * 4 + valueD * 8, ti.i32)
            for k in ti.static(range(2)):  # [0,2]
                if self.trangle_table[self.mc_result[i, j]][2 * k] != -1:
                    n = ti.atomic_add(self.edge_num[None], 1)  # 边的个数 （note 并没有和单元绑定）

                    self.edge[n].begin_point = self.gen_edge_pos(i, j, self.trangle_table[self.mc_result[i, j]][
                        2 * k])  # 表示边的起点在当前网格单元的位置。
                    self.edge[n].end_point = self.gen_edge_pos(i, j, self.trangle_table[self.mc_result[i, j]][
                        1 + 2 * k])  # 表示边的终点。

    # --------------------------------------------------表面张力---------------------------------------------

    # 计算梯度算子（法线）(在表面网格上)
    @ti.kernel
    def calculate_gradient(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j = I
            u, v = .0, .0
            # 根据不同边界条件，给出网格点的SDF的梯度
            if i == 0:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i, j]) * 0.5 * self.diff_inv_dx
            elif i == self.diff_n_grid - 1:
                u = (self.sign_distance_field[i, j] - self.sign_distance_field[i - 1, j]) * 0.5 * self.diff_inv_dx
            else:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i - 1, j]) * 0.5 * self.diff_inv_dx

            if j == 0:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j]) * 0.5 * self.diff_inv_dx
            elif j == self.diff_n_grid - 1:
                v = (self.sign_distance_field[i, j] - self.sign_distance_field[i, j - 1]) * 0.5 * self.diff_inv_dx
            else:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j - 1]) * 0.5 * self.diff_inv_dx
            self.SDF[I].gradient = ti.Vector([u, v]).normalized()       # 上下左右相邻的网格SDF梯度

    # 计算拉普拉斯算子（曲率）
    @ti.kernel
    def calculate_laplacian(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j = I
            u, v = .0, .0
            # 处理 i 方向上的拉普拉斯算子
            if i == 0:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i, j]) * self.diff_inv_dx **2
            elif i == self.diff_n_grid - 1:
                u = (-self.sign_distance_field[i, j] + self.sign_distance_field[i - 1, j]) * self.diff_inv_dx **2
            else:
                u = (self.sign_distance_field[i + 1, j] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i - 1, j]) * self.diff_inv_dx **2
            # 处理 j 方向上的拉普拉斯算子
            if j == 0:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j]) * self.diff_inv_dx **2
            elif j == self.diff_n_grid - 1:
                v = (-self.sign_distance_field[i, j] + self.sign_distance_field[i, j - 1]) * self.diff_inv_dx **2
            else:
                v = (self.sign_distance_field[i, j + 1] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i, j - 1]) * self.diff_inv_dx **2
            # 计算拉普拉斯算子
            self.SDF[I].laplacian = u + v


    @ti.kernel
    def init_surface_particles(self):                                       # 每帧开始，将构建的表面粒子删除
        for i in range(self.surface_particle_num[None]):
            self.surface_particles[i].position = ti.Vector([0.0, 0.0])
        self.surface_particle_num[None] = 0


    @ti.kernel
    def create_particle(self):                                          # 根据edge容器生成表面节点坐标，并赋予给容器
        self.surface_particle_num[None] = 0

        for n in range(self.edge_num[None]):
            ab = self.edge[n].end_point - self.edge[n].begin_point      # 终点—起点坐标
            for i in ti.static(range(4)):
                pos = self.edge[n].begin_point + ((i / 3) * ab)         # 离散出更多点的坐标
                index = ti.atomic_add(self.surface_particle_num[None], 1)   # 计算所有表面粒子数量
                self.surface_particles[index].position = pos                # 储存表面粒子坐标

    @ti.func
    def linear_interpolation(self, pos: ti.template()):          # 线性插值函数：对周围四个网格角点的SDF梯度和拉普拉斯算子进行加权
        base = (pos * self.diff_inv_dx).cast(int)
        fx = pos * self.diff_inv_dx - base.cast(float)
        w = [(1 - fx) * self.diff_dx, fx * self.diff_dx]
        result_g = ti.Vector([0.0, 0.0])
        result_l = 0.0

        for i, j in ti.static(ti.ndrange(2, 2)):                # 线性插值：网格SDF梯度和算子 ——> 表面粒子
            weight = w[i][0] * w[j][1] * self.diff_inv_dx ** 2
            offset = ti.Vector([i, j])
            result_g += self.SDF[base + offset].gradient * weight
            result_l += self.SDF[base + offset].laplacian * weight
        return result_g, result_l


    def build_surface(self, simulator):                                      # 表面建立
        self.gen_level_set(simulator.p_x, simulator.p_num[None])          # 隐式液表——网格位置符号表示液面
        self.implicit_to_explicit()                                          # 显式液表——由具体的坐标表示液面

        if self.consider_tension:
            self.init_surface_particles()                   # 初始化表面粒子
            self.calculate_gradient()                       # 计算SDF梯度
            self.calculate_laplacian()                      # 计算SDF曲率
            self.create_particle()                          # 根据显式液面线段生成表面粒子坐标
