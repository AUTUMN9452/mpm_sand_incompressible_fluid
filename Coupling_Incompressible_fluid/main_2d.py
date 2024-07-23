from math import pi
import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)
from coupling import CouplingSandWater
from visualize_2d import *

ti.init(arch=ti.gpu)
# 定义颜色数组（三个颜色）

if __name__ == '__main__':

    coupling = CouplingSandWater()
    coupling.initialize()
    # 后处理显示
    step = 0
    gui = ti.GUI("Taichi incompressible", res=512, background_color = 0xFFFFFF)

    color_array = ti.Vector.field(3, dtype=ti.f32, shape=2)
    color_array[0] = [0.7, 0.7, 0.4]  #
    color_array[1] = [1.0, 0.0, 0.0]  #
    visualizor = DataColorMapper(coupling.sand_sim.n_particle_s[None], color_array, res = (coupling.sand_sim.n_grid, coupling.sand_sim.n_grid))

    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):

        for s in range(50):  # 总时间帧
            coupling.substep()

        gui.circles(coupling.water_sim.p_x.to_numpy()[:coupling.water_sim.p_num[None]], radius=1.2, color=0x3655DA)

        # min_value = min(coupling.sand_sim.k_f.to_numpy()[:coupling.sand_sim.n_particle_s[None]])

        visualizor.update_color_field(coupling.sand_sim.saturation, visual_range=(0, +1))  # 可视化饱和度
        gui.circles(coupling.sand_sim.x_s.to_numpy()[:coupling.sand_sim.n_particle_s[None]], radius=1.1, color=visualizor.color.to_numpy())

        # gui.circles(coupling.sand_sim.x_s.to_numpy()[:coupling.sand_sim.n_particle_s[None]], radius=1.2, color=0xEAB226)


        gui.show()

