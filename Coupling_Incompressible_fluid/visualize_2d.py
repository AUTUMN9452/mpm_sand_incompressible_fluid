import taichi as ti
import numpy as np

@ti.data_oriented
class DataColorMapper:
    def __init__(self, num_particles, color_array, dim = 2, res = (100,100)):
        self.num_particles = num_particles
        self.color = ti.field(dtype=ti.i32, shape=num_particles)
        self.color_array = color_array
        self.res = res
        self.dim = dim
        self.index_boundary = ti.Vector.field(self.dim, dtype=ti.f32, shape = self.res[0]**2)
        self.color_boundary = ti.field(dtype=int, shape = self.res[0]**2)

    @ti.func
    def lerp_color(self, c1, c2, t):
        return c1 * (1 - t) + c2 * t

    @ti.func
    def rgb_to_hex(self, rgb):
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        return (r << 16) + (g << 8) + b


    @ti.kernel
    def update_color_vector(self, data: ti.template(),dimension: int, visual_range: ti.template()):
        max_data = visual_range[1]
        min_data = visual_range[0]

        for i in range(self.num_particles):
            normalized_value = ti.max(0,ti.min((data[i][dimension]-min_data) / (max_data-min_data),1))  # (0-1)
            # print(normalized_value)
            color_index = normalized_value * (self.color_array.shape[0] - 1)                # 颜色条范围
            idx0 = int(ti.floor(color_index))                                               # 左侧颜色编号
            idx1 = min(idx0 + 1, self.color_array.shape[0] - 1)                             # 右侧颜色编号
            t = color_index - idx0
            rgb_color  = self.lerp_color(self.color_array[idx0], self.color_array[idx1], t)
            self.color[i] = self.rgb_to_hex(rgb_color)


    @ti.kernel
    def update_color_field(self, data: ti.template(), visual_range: ti.template()):
        max_data = visual_range[1]
        min_data = visual_range[0]

        for i in range(self.num_particles):
            normalized_value = ti.max(0,ti.min((data[i]-min_data) / (max_data-min_data),1))  # (0-1)
            # print(normalized_value)
            color_index = normalized_value * (self.color_array.shape[0] - 1)                # 颜色条范围
            idx0 = int(ti.floor(color_index))                                               # 左侧颜色编号
            idx1 = min(idx0 + 1, self.color_array.shape[0] - 1)                             # 右侧颜色编号
            t = color_index - idx0
            rgb_color  = self.lerp_color(self.color_array[idx0], self.color_array[idx1], t)
            self.color[i] = self.rgb_to_hex(rgb_color)


    @ti.kernel
    def update_boundary_color(self, data: ti.template()):
        for I in ti.grouped(data):
            if data[I] == 2:
                index = I[0] + I[1] * self.res[0]
                self.color_boundary[index] = 0xD3D3D3  # 固体为灰色
                self.index_boundary[index] = ti.Vector([(float(I[0]) + 0.5) / self.res[0], (float(I[1]) + 0.5) / self.res[1]])  # 网格位置
