import taichi as ti
from mgpcg import MGPCGPoissonSolver

@ti.data_oriented
class PressureProjectStrategy:
    def __init__(self, dim, velocity, p0):
        self.dim = dim
        self.velocity = velocity
        self.p0 = p0 # the standard atmospheric pressure


    @ti.kernel
    def build_b_kernel(self, 
                       cell_type : ti.template(), 
                       b : ti.template()):
        for I in ti.grouped(cell_type):                   # 遍历所有网格单元
            if cell_type[I] == 1:                         # 如果网格单元是流体
                for k in ti.static(range(self.dim)):      # 遍历每个维度
                    offset = ti.Vector.unit(self.dim, k)  # 获取在当前维度上的单位向量 [1,0] and [0,1]
                    b[I] += (self.velocity[k][I] - self.velocity[k][I + offset])  # v[0][i,j] - v[0][i+1,j] + v[1][i,j] - v[1][i,j+1]
                b[I] *= self.scale_b                                              # scale_b = 1/dx 网格间距的导数——>b为速度散度

        for I in ti.grouped(cell_type):                   # 处理与固体和空气交界
            if cell_type[I] == 1:
                for k in ti.static(range(self.dim)):      # 遍历每个维度
                    for s in ti.static((-1, 1)):          # 在当前维度的正负方向上进行处理——>计算速度
                        offset = ti.Vector.unit(self.dim, k) * s    # [1,0], [0,1],[-1,0], [0,-1] 上下左右四个方格
                        if cell_type[I + offset] == 2:    # 与固体接触
                            if s < 0: b[I] -= self.scale_b * (self.velocity[k][I] - 0)
                            else: b[I] += self.scale_b * (self.velocity[k][I + offset] - 0)     # 减去与固体交界的面速度部分，即固体速度为0
                        elif cell_type[I + offset] ==0:                                # 与空气接触
                            b[I] += self.scale_A * self.p0                       # 标准大气压 scale_A = dt / (rho * dx **2) ?
                                
    def build_b(self, solver : MGPCGPoissonSolver):
        self.build_b_kernel(solver.grid_type[0], 
                            solver.b)

    @ti.kernel
    def build_A_kernel(self, 
                       level : ti.template(),
                       grid_type : ti.template(), 
                       Adiag : ti.template(), 
                       Ax : ti.template()):
        for I in ti.grouped(grid_type):
            if grid_type[I] == 1:                                             # 目标网格为流体
                for k in ti.static(range(self.dim)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(self.dim, k) * s                        # [1,0], [0,1],[-1,0], [0,-1] 上下左右四个方格
                        if grid_type[I + offset] == 1:                        # 如果周围是流体
                            Adiag[I] += self.scale_A                                    # 对角矩阵代表当前单元 I 的系数
                            if ti.static(s > 0):                                        # 并且是在前方??? todo 为什么这里只考虑一个方向的相邻位置系数？
                                                                                        # todo 答：这个意思是表示右侧和上侧是否具有流体网格，和下面代码的搭配使用
                                                                                        # todo 答：ret += Ax[I - offset][i] * x[I - offset] + Ax[I][i] * x[I + offset]
                                                                                        # todo 答：           左侧有无网格   * 左侧网格压力     +右侧是否有网格  * 右侧网格压力
                                Ax[I][k] = -self.scale_A                                # 非对角矩阵代表当前单元在方向 k上的相邻单元的系数
                        elif grid_type[I + offset] == 0:                        # 如果周围是空气
                            Adiag[I] += self.scale_A                                # scale_A = dt / (rho * dx **2)


    def build_A(self, solver : MGPCGPoissonSolver, level):
        self.build_A_kernel(level, 
                            solver.grid_type[level], 
                            solver.Adiag[level], 
                            solver.Ax[level])
'''
[-1, 0] s =  -1 k =  0
[1, 0]  s =  1  k =  0
[0, -1] s =  -1 k =  1
[0, 1]  s =  1  k =  1'''

