import numpy as np

D2R = np.pi / 180.0


# 步长推算
# 目前的方法不具备科学性，临时使用
def step_stride(max_acceleration, min_acceleration):
    return np.power(abs(max_acceleration - min_acceleration), 1 / 4) * 0.66


# 步数探测 用于批量处理
def step_detection(acc_norm, frequency=50):
    # offset = frequency / 100
    # slide = 60 * offset  # 滑动窗口（100Hz的采样数据）
    max_acceleration = 0.04  # np.mean(acc_norm) + 0.1
    min_acceleration = -0.1  # np.mean(acc_norm) + 0.1
    min_interval = 0.3
    interval = int(0.3 * frequency)
    steps = []
    peak = []  # 索引和速度
    peak_update = []
    valley = []  # 谷值
    valley_update = []
    # 检测峰值
    for i in range(interval, acc_norm.shape[0] - 2):
        if acc_norm[i - 1] > acc_norm[i - 2] and acc_norm[i - 1] > acc_norm[i] \
                and acc_norm[i - 1] > max_acceleration:
            peak.append([i - 1, acc_norm[i - 1, 0]])
    peak = np.array(peak)
    # 去除异常峰值
    last_peak = peak[0, :]
    for i in range(1, peak.shape[0]):
        if peak[i, 0] - last_peak[0] > interval:
            peak_update.append(last_peak)
            last_peak = peak[i, :]
        else:
            last_peak = last_peak if peak[i, 1] < last_peak[1] else peak[i, :]
        if i == peak.shape[0] - 1:
            peak_update.append(last_peak)

    peak_update = np.array(peak_update)
    # 检测谷值
    for i in range(2 * interval, acc_norm.shape[0] - 2):
        if acc_norm[i - 1] < acc_norm[i - 2] and acc_norm[i - 1] < acc_norm[i] \
                and acc_norm[i - 1] < min_acceleration:
            valley.append([i - 1, acc_norm[i - 1, 0]])

    valley = np.array(valley)
    # print(peak.shape)
    # 去除异常谷值
    last_valley = valley[0, :]
    for i in range(1, valley.shape[0]):
        if valley[i, 0] - last_valley[0] > interval:
            valley_update.append(last_valley)
            last_valley = valley[i, :]
        else:
            last_valley = last_valley if valley[i, 1] > last_valley[1] else valley[i, :]
        if i == valley.shape[0] - 1:
            valley_update.append(last_valley)

    valley_update = np.array(valley_update)
    p_index = 0
    v_index = 0
    flag = -1
    peak_valley = []
    s_index = 0
    # 把峰值和谷值放一起

    for i in range(peak_update.shape[0] + valley_update.shape[0]):
        if peak_update[p_index, 0] < valley_update[v_index, 0]:
            peak_valley.append([peak_update[p_index, 0], peak_update[p_index, 1]])
            p_index += 1
        else:
            peak_valley.append([valley_update[v_index, 0], valley_update[v_index, 1]])
            v_index += 1
        if p_index == peak_update.shape[0] or v_index == valley_update.shape[0]:
            break

    peak_valley = np.array(peak_valley)

    # 检测脚步
    steps = []  # 开始时间，结束时间，开始的加速度，结束的加速度
    i = 1
    while (i < peak_valley.shape[0]):
        if peak_valley[i, 1] * peak_valley[i - 1, 1] < 0:
            steps.append([peak_valley[i - 1, 0], peak_valley[i, 0],
                          peak_valley[i - 1, 1], peak_valley[i, 1]])
            i += 1
        i += 1

    steps = np.array(steps)
    # import matplotlib.pyplot as plt
    # t = np.mat(np.arange(0, acc_norm.shape[0]) * 0.02).T
    # t_peak = choose_from(t, steps[:, 0])
    # t_val = choose_from(t, steps[:, 1])
    # fig_1, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(t, acc_norm, 'b', label='acc_norm')
    # ax.plot(t_peak, steps[:, 2], 'rx', label='peak')
    # ax.plot(t_val, steps[:, 3], 'gx', label='valley')
    # ax.legend(loc=2)
    # ax.set_xlabel('t/s')
    # ax.set_ylabel('m/s')
    # ax.set_title('step_show')
    # plt.show()
    return steps


# 平滑数据
def smooth_data(da, sample_rate):
    da_sm = np.zeros(da.shape[0])
    for i in range(da.shape[0]):
        if i >= sample_rate - 1:
            da_sm[i] = np.mean(da[i + 1 - sample_rate:i + 1])
        else:
            da_sm[i] = np.mean(da[0:i + 1])
    return np.array(da_sm)


# 计算roll和pitch
def cal_roll_pitch(ax, ay, az):
    roll = np.arctan2(-ay, -az)
    pitch = np.arctan2(ax, np.sqrt(ay ** 2 + az ** 2))
    return roll, pitch


# 航向角积分量
def cal_yaw(gry, roll, pitch):
    cosr = np.cos(roll)
    cosp = np.cos(pitch)
    sinr = np.sin(roll)
    sinp = np.sin(pitch)

    yaw_diff = gry[0, 1] * sinr / cosp + gry[0, 2] * cosr / cosp

    return yaw_diff


# 保证角度在-pi到pi
def central_yaw(yaw):
    while yaw > np.pi or yaw <= -np.pi:
        if yaw > np.pi:
            yaw = yaw - np.pi * 2
        elif yaw <= -np.pi:
            yaw = yaw + np.pi * 2
    return yaw


# 从给定的索引当中选数据
def choose_from(array, target):
    res = np.zeros((target.shape[0], array.shape[1]))
    for i in range(target.shape[0]):
        res[i, :] = array[int(target[i]), :]
    return np.array(res)


# pdr算法
def pdr(t_steps, init_pos, yaw_steps, step_length):
    e1 = init_pos[0]
    n1 = init_pos[1]
    # pos = np.zeros((t_steps.shape[0], 4))  # t, e, n, heading

    t1 = t_steps
    e1 += (step_length * np.sin(yaw_steps))
    n1 += (step_length * np.cos(yaw_steps))
    pos = np.array([t1, e1, n1, yaw_steps, step_length])

    return pos


# 删除离群点
def del_outlier(scope, data):
    for i in range(scope, data.shape[0]):
        std_x = np.std(data[i - scope:i + 1, 0])
        std_y = np.std(data[i - scope:i + 1, 1])
        mean_x = np.mean(data[i - scope:i + 1, 0])
        mean_y = np.mean(data[i - scope:i + 1, 1])
        for j in range(scope + 1):
            if abs(data[i - j, 0] - mean_x) + abs(data[i - j, 1] - mean_y) > 4:
                if j == 0:
                    data[i, :2] = data[i - 1, :2] + (data[i - 1, :2] - data[i - 2, :2])
                else:
                    data[i - j, :2] = (data[i - j - 1, :2] + data[i - j + 1, :2]) / 2

    return data
