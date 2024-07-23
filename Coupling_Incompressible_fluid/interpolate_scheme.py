import taichi as ti

@ti.func
def splat_vp_apic(dx, data, weights, pos, v, c, stagger, mass):  # 将速度
    base = (pos /dx - (stagger + 0.5)).cast(ti.i32)
    fx = pos /dx - (base.cast(ti.f32) + stagger)

    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(ti.f32) - fx) * dx
            weight = w[i][0] * w[j][1]
            data[base + offset] += weight * mass * (v + c.dot(dpos))  # 仿射速度
            weights[base + offset] += weight * mass  # 权重累加


@ti.func
def sample_vp_apic(dx, data, pos, stagger):  # 通过相邻的四个点——对中间点的数据进行插值计算

    base = (pos /dx - (stagger + 0.5)).cast(ti.i32)  # 计算粒子所在网格的基准点——整型
    fx = pos /dx - (base.cast(ti.f32) + stagger)  # 计算粒子在网格单元中的相对位置
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline
    v_pic = 0.0
    for i, j in ti.static(ti.ndrange(3, 3)):  # 对3x3网格开始历遍，每个粒子都需要对该3x3网格贡献
        offset = ti.Vector([i, j])
        weight = w[i][0] * w[j][1]
        v_pic += weight * data[base + offset]
    return v_pic


@ti.func
def sample_cp_apic(dx, grid_v, xp, stagger):  # 粒子的仿射速度
    base = (xp /dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp /dx - (base.cast(ti.f32) + stagger)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline
    # w_grad = [fx - 1.5, -2 * (fx - 1), fx - 3.5]                            # Bspline gradient
    w_grad = [fx - 1.5, -2 * (fx - 1), fx - 0.5]                              # Bspline gradient
    cp = ti.Vector([0.0, 0.0])

    for i, j in ti.static(ti.ndrange(3, 3)):                                  # 对3x3网格开始历遍，每个粒子都需要对该3x3网格贡献
        offset = ti.Vector([i, j])
        weight_grad = ti.Vector([w_grad[i][0] * w[j][1]/dx, w[i][0] * w_grad[j][1]/dx])
        cp += weight_grad * grid_v[base + offset]
    return cp

