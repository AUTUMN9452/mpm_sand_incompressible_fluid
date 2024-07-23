import taichi as ti
import numpy as np

@ti.data_oriented
class Initializer2D:
    def __init__(self,dim = 2, particle_diameter = 0.2):
        self.dim = dim
        self.particle_diameter = particle_diameter
        self.FLUID = 1
        self.SOLID = 2

    @ti.kernel
    def init_boundary(self,cell_type : ti.template(), res: ti.template()):          # 将边界的单元设置成固体
        for I in ti.grouped(cell_type):
            if any(I <= 2) or any(I >= res[0] - 3):                                 # 边界附近的三个单元格都是固体
                cell_type[I] = 2                                                    #

    @ti.kernel
    def add_particles(self,         particle_num : ti.template(),                   # 向粒子系统添加新的粒子位置、确定材质。
                                    position: ti.types.ndarray(),
                                    p_num: ti.template(),
                                    p_x: ti.template()):

        for idx in range(p_num[None], p_num[None] + particle_num):                  # 遍历要添加的粒子的索引范围
            relative_idx = idx - p_num[None]                                        # 计算相对索引
            pos = ti.Vector.zero(ti.f32, self.dim)                                  # 维度为 3，存储位置信息
            for dim_idx in ti.static(range(self.dim)):
                pos[dim_idx] = position[relative_idx, dim_idx]                      # 将位置数据赋值给向量的相应维度
            p_x[idx] = pos
        p_num[None] += particle_num                                              # 更新粒子系统中已分配内存的粒子数，增加新添加的粒子数


    def point_in_polygon(self, point, polygon):  # 判断是否在多边形区域内
        count = 0
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            if ((p1[1] > point[1]) != (p2[1] > point[1])) and \
                    (point[0] < (p2[0] - p1[0]) * (point[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]):
                count += 1
        return count % 2 == 1


    def set_block(self,simulator, polygon):                                         # todo 生成多边形固体区域
        I = simulator.cell_type.shape
        min_x = min(vertex[0] for vertex in polygon)
        max_x = max(vertex[0] for vertex in polygon)
        min_y = min(vertex[1] for vertex in polygon)
        max_y = max(vertex[1] for vertex in polygon)
        for i in range(max(0, int(np.floor(min_x))), min(I[0], int(np.ceil(max_x)))):
            for j in range(max(0, int(np.floor(min_y))), min(I[1], int(np.ceil(max_y)))):
                if self.point_in_polygon((i, j), polygon):
                    simulator.cell_type[i, j] = 2                                   # 将矩形区域内的值设为固体


    def add_polygon(self, particle_num, pos, polygon):                # 根据角点生成多面体
        # 计算多边形的包围盒（bounding box）
        min_x = min(vertex[0] for vertex in polygon)
        max_x = max(vertex[0] for vertex in polygon)
        min_y = min(vertex[1] for vertex in polygon)
        max_y = max(vertex[1] for vertex in polygon)
        grid_spacing = self.particle_diameter
        position_list = []
        # 遍历包围盒内的每个网格点
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                point = np.array([x, y], dtype=np.float32)
                if self.point_in_polygon(point, polygon):
                    position_list.append(point)
                y += grid_spacing
            x += grid_spacing
        position_arr = np.array(position_list, dtype=np.float32)
        self.add_particles(len(position_list), position_arr, particle_num, pos)


    def add_polygon_random(self,max_num, particle_num, pos, polygon):
        # 计算多边形的包围盒（bounding box）
        min_x = min(vertex[0] for vertex in polygon)
        max_x = max(vertex[0] for vertex in polygon)
        min_y = min(vertex[1] for vertex in polygon)
        max_y = max(vertex[1] for vertex in polygon)

        position_list = []
        # 使用蒙特卡洛方法在多边形内随机生成粒子
        while len(position_list) < max_num:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = np.array([x, y], dtype=np.float32)
            if self.point_in_polygon(point, polygon):
                position_list.append(point)

        position_arr = np.array(position_list, dtype=np.float32)
        self.add_particles(len(position_list), position_arr, particle_num, pos)


    def add_cyclic(self, center, radius, particle_num, pos):                # 根据圆心和半径
        grid_spacing = self.particle_diameter
        position_list = []

        min_x, max_x = center[0] - radius, center[0] + radius
        min_y, max_y = center[1] - radius, center[1] + radius

        # 遍历包围盒内的每个网格点
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                point = np.array([x, y], dtype=np.float32)
                distance = np.linalg.norm(point - center)
                if distance <= radius:
                    position_list.append(point)
                y += grid_spacing
            x += grid_spacing
        position_arr = np.array(position_list, dtype=np.float32)
        self.add_particles(len(position_list), position_arr, particle_num, pos)


    def init_scene_sand(self, simulator):  # 长方形流体区域
        # polygon = np.array([[4,0.4], [6,0.4], [6, 4],[4, 4]], dtype=np.float32)    # dam break
        polygon = np.array([[0.3, 0.03], [0.78, 0.03], [0.62, 0.2], [0.46, 0.2]], dtype=np.float32)    # dam break
        # polygon = np.array([[0.4, 0.03], [0.9, 0.03], [0.75, 0.2], [0.55, 0.2]], dtype=np.float32)  # dam break

        # self.set_block(simulator, ((50,0), (100,0), (100, 20)))                         # 多边形边界
        self.add_polygon(simulator.n_particle_s, simulator.x_s, polygon)  # 随机生成


    def init_scene_water(self, simulator):  # 长方形流体区域
        self.init_boundary(simulator.cell_type, simulator.res)

        polygon = np.array([[0.04,0.04], [0.31,0.04], [0.46, 0.2],[0.46, 0.3],[0.04, 0.3]], dtype=np.float32)    # dam break
        self.add_polygon(simulator.p_num, simulator.p_x, polygon)                   # 均匀生成
        # self.add_cyclic([0.45,0.45], 0.15, simulator.p_num, simulator.p_x)                                       # 圆形区域
