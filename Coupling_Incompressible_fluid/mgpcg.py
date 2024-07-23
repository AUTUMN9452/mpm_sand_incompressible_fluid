import taichi as ti

@ti.data_oriented
class MGPCGPoissonSolver:
    def __init__(self, dim, res, n_mg_levels = 4, pre_and_post_smoothing = 2, bottom_smoothing = 50, real = float):

        self.FLUID = 1
        self.SOLID = 2
        self.AIR = 0

        # grid parameters
        self.dim = dim
        self.res = res
        self.n_mg_levels = n_mg_levels                                  # 共四层网格，从稀到密
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing
        self.real = real

        # rhs of linear system
        self.b = ti.field(dtype=real, shape=res) # Ax=b

        self.r = [ti.field(dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # residual
        self.z = [ti.field(dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # M^-1 self.r (z是多重预条件化后的残差向量)

        # grid type
        self.grid_type = [ti.field(dtype=ti.i32, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)]

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [ti.field(dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # A(i,j,k)(i,j,k)
        self.Ax = [ti.Vector.field(dim, dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # Ax=A(i,j,k)(i+1,j,k), Ay=A(i,j,k)(i,j+1,k), Az=A(i,j,k)(i,j,k+1)
        
        
        self.x = ti.field(dtype=real, shape=res) # solution
        self.p = ti.field(dtype=real, shape=res) # conjugate gradient
        self.Ap = ti.field(dtype=real, shape=res) # matrix-vector product
        self.sum = ti.field(dtype=real, shape=()) # storage for reductions
        self.alpha = ti.field(dtype=real, shape=()) # step size
        self.beta = ti.field(dtype=real, shape=()) # step size

    @ti.kernel
    def init_gridtype(self, grid0 : ti.template(), grid : ti.template()):  # 初始化稀疏后的网格类型——空气、流体、固体？
        for I in ti.grouped(grid):                                    # 循环遍历所有网格单元
            I2 = I * 2                                                # 计算当前网格单元的索引乘以2，这是因为 grid0 的网格尺寸是 grid 的两倍。
            tot_fluid = 0                                             # 初始化两个计数器，用于计算当前网格单元中的流体单元和空气单元的数量。
            tot_air = 0
            for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):    # 四个子单元[0,1],[1,1],[0,0],[1,0]
                attr = int(grid0[I2 + offset])                        # 获取子单元的属性识别符
                if attr == self.AIR: tot_air += 1
                elif attr == self.FLUID: tot_fluid += 1
            if tot_air > 0: grid[I] = self.AIR                        # 有空气即为空气（最高优先级）
            elif tot_fluid > 0: grid[I] = self.FLUID                  # 设为液体
            else: grid[I] = self.SOLID                                # 设为固体

    
    @ti.kernel
    def initialize(self):
        for I in ti.grouped(ti.ndrange(* [self.res[_] for _ in range(self.dim)])):
            self.r[0][I] = 0                                          # 残差
            self.z[0][I] = 0                                          #
            self.Ap[I] = 0
            self.p[I] = 0
            self.x[I] = 0
            self.b[I] = 0

        for l in ti.static(range(self.n_mg_levels)):                  # 初始化多重网格每一层的网格类型（共四层，用于迭代）
            for I in ti.grouped(ti.ndrange(* [self.res[_] // (2**l) for _ in range(self.dim)])):
                self.grid_type[l][I] = 0
                self.Adiag[l][I] = 0
                self.Ax[l][I] = ti.zero(self.Ax[l][I])

    def reinitialize(self, cell_type, strategy):                      # 重新初始化各种变量，构建泊松方程的系数矩阵 A 和源向量 b
        self.initialize()                                             #
        self.grid_type[0].copy_from(cell_type)                        # 将最浅层的网格属性进行拷贝至 第一重MGPCG迭代容器中
        strategy.build_b(self)
        strategy.build_A(self, 0)

        for l in range(1, self.n_mg_levels):                          # 构建每一重网格的材料属性，以及系数矩阵A
            self.init_gridtype(self.grid_type[l - 1], self.grid_type[l])
            strategy.build_A(self, l)

    def full_reinitialize(self, strategy):
        self.initialize()
        strategy.build_b(self)
        for l in range(self.n_mg_levels):
            self.grid_type[l].fill(1)
            strategy.build_A(self, l)

    @ti.func
    def neighbor_sum(self, Ax, x, I):                               # 计算周围网格的贡献
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):                        # 历遍不同维度
            offset = ti.Vector.unit(self.dim, i)                    # [1,0], [0,1]
            ret += Ax[I - offset][i] * x[I - offset] + Ax[I][i] * x[I + offset]  # 相邻左右单元对ret的贡献？
        return ret

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):       # 迭代求解 z ->表示不同层级上的解
        # phase = red/black Gauss-Seidel phase 红黑 Gauss-Seidel 迭代法
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase and self.grid_type[l][I] == self.FLUID:       # 判断容器编号的奇偶性以及是否是流体
                self.z[l][I] = (self.r[l][I] - self.neighbor_sum(self.Ax[l], self.z[l], I)) / self.Adiag[l][I]  # 使得z逼近与真实解x

    @ti.kernel
    def restrict(self, l: ti.template()):                           # 将残差从细网格投射到粗网格的过程：细 ——> 粗
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == self.FLUID:
                Az = self.Adiag[l][I] * self.z[l][I]                # 代表了每个单元自身对解的影响
                Az += self.neighbor_sum(self.Ax[l], self.z[l], I)   # 邻近粒子对解的影响
                res = self.r[l][I] - Az
                self.r[l + 1][I // 2] += res                        # 将残差限制到下一层网格，累加至粗糙网格中

    @ti.kernel
    def prolongate(self, l: ti.template()):                         # 将较粗网格的校正值传播到较细网格：粗 ——> 细
        for I in ti.grouped(self.z[l]):                             # 将粗网格上的校正值传播回细网格，从而修正细网格上的解
            self.z[l][I] += self.z[l + 1][I // 2]

    def v_cycle(self):
        self.z[0].fill(0.0)                                     # 初始化最细网格层的校正值为0
        for l in range(self.n_mg_levels - 1):                   # 最细网格层——倒数第二层的每一层
            for i in range(self.pre_and_post_smoothing):        # 预平滑（pre-smoothing）
                self.smooth(l, 0)
                self.smooth(l, 1)                               # 实现红黑 Gauss-Seidel 平滑

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)                                    # 残差限制到较粗层——>残渣已经传递到了最粗层

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing // 2):             # 求解最粗网格解
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):         # 将最粗网格的解往细网格传：粗 ——> 细
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):        # 后平滑
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,                                             # 多重网格预条件共轭梯度法（MGPCG）来求解线性方程组Ax = b
              max_iters=-1,
              verbose=True,                                    # 是否输出详细信息
              rel_tol=1e-12,                                    # 相对误差容忍度
              abs_tol=1e-14,                                    # 绝对误差容忍度
              eps=1e-12):                                       # 用于防止除零

        self.r[0].copy_from(self.b)                             # 原始网格的初始速度散度 通过build_b计算
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]                            # 初始残差的内积 r^{T}·r

        if verbose:
             print(f"init rtr = {initial_rTr}")                 #

        tol = max(abs_tol, initial_rTr * rel_tol)               # 计算收敛容忍度
        # 第一步的情况：
        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        self.v_cycle()                                          # 第一次多网格迭代，进入多重网格的循环后出来
        self.update_p()                                         # 更新 p

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # Conjugate gradients
        iter = 0
        while max_iters == -1 or iter < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)                    # 步长 a

            # self.x = self.x + self.alpha self.p
            # self.r = self.r - self.alpha self.Ap
            self.update_xr()                                            # 更新残差 r 和解 x
            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            # if verbose:
            #     print(f'iter {iter}, |residual|_2={ti.sqrt(rTr)}')

            if rTr < tol:                                               # 当误差足够小破除循环
                break

            # self.z = M^-1 * self.r (z是多重预条件化后的残差向量)
            self.v_cycle()                                              # 进行一遍从 细网格->粗网格->细网格 的求解过程，获得更小的 r

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)     # 迭代步长：B(k+1) = r(k+1).transpose()·r(k+1) / r(k).transpose()·r(k)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            iter += 1

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            if self.grid_type[0][I] == self.FLUID:          # 仅处理网格类型为 FLUID 的单元
                self.sum[None] += p[I] * q[I]               # 将 p[I] 和 q[I] 的乘积累加到 sum 中

    @ti.kernel
    def compute_Ap(self):                                   # 计算共轭梯度法中的Ap
        for I in ti.grouped(self.Ap):
            if self.grid_type[0][I] == self.FLUID:
                r = self.Adiag[0][I] * self.p[I]
                r += self.neighbor_sum(self.Ax[0], self.p, I)
                self.Ap[I] = r

    @ti.kernel
    def update_xr(self):
        alpha = self.alpha[None]
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == self.FLUID:
                self.x[I] += alpha * self.p[I]              # 更新解 x(k+1) = x(k) + a(k)·p(k)
                self.r[0][I] -= alpha * self.Ap[I]          # 更新残差 r(k+1) = r(k) - a(k)·A·p(k)

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == self.FLUID:          #
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]      # 更新 P(k+1) = r(k+1) - B(k)·p(k)
