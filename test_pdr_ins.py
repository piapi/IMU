import numpy as np
import matplotlib.pyplot as plt
import file_test.readfile as rd
import IMU_Process as imu
import UWB.dis2pos as uwb
import uwb_pdr_fusion as fusion

D2R = np.pi / 180.0
init_yaw = 73 * D2R
init_pos = np.array([3.08, 6.22])


# is_from_uwb这个数据是不是来着于UWB设备的IMU的

def test_pdr(is_from_uwb, file):
    da_sensor = rd.read_file(file)
    t = da_sensor[:, 0] if not is_from_uwb else da_sensor[:, 0] / 1000
    gx = np.array(da_sensor[:, 1]) * D2R
    gy = np.array(da_sensor[:, 2]) * D2R
    gz = np.array(da_sensor[:, 3]) * D2R
    gry = da_sensor[:, 1:4] * D2R if not is_from_uwb else da_sensor[:, 1:4]
    ax = da_sensor[:, 4]
    ay = da_sensor[:, 5]
    az = da_sensor[:, 6]
    acc = da_sensor[:, 4:7]

    sample_rate = round(t.shape[0] / (t[-1, 0] - t[0, 0]))

    ax_sm = imu.smooth_data(ax, sample_rate)
    ay_sm = imu.smooth_data(ay, sample_rate)
    az_sm = imu.smooth_data(az, sample_rate)

    acc_norm = np.zeros(t.shape)
    for i in range(t.shape[0]):
        acc_norm[i] = np.sqrt(ax_sm[i] ** 2 + ay_sm[i] ** 2 + az_sm[i] ** 2) - 9.8

    steps = imu.step_detection(acc_norm, sample_rate)

    stepsLen = np.zeros((steps.shape[0], 2))  # 步长
    for i in range(steps.shape[0]):
        stepsLen[i, 0] = steps[i, 1]
        stepsLen[i, 1] = imu.step_stride(steps[i, 2], steps[i, 3])

    print(steps.shape[0])
    roll = np.zeros(t.shape[0])
    pitch = np.zeros(t.shape[0])
    yaw = np.zeros((t.shape[0], 1))
    for i in range(t.shape[0]):
        roll[i], pitch[i] = imu.cal_roll_pitch(ax_sm[i], ay_sm[i], az_sm[i])
        if i == 0:
            yaw[i] = init_yaw
        else:
            yaw[i] = yaw[i - 1] + imu.cal_yaw(gry[i, :], roll[i], pitch[i]) * (t[i, 0] - t[i - 1, 0])
        yaw[i] = imu.central_yaw(yaw[i])

    t_step = imu.choose_from(t, steps[:, 1])
    t_peak = imu.choose_from(t, steps[:, 0])
    yaw_step = imu.choose_from(yaw, steps[:, 1])  # if not uwb else imu.choose_from(da_sensor[:, -1], steps[:, 0]) * D2R
    pos = np.zeros((t_step.shape[0], 5))
    pos_0 = init_pos
    for i in range(t_step.shape[0]):
        pos[i, :] = imu.pdr(t_step[i], pos_0, yaw_step[i, 0], stepsLen[i, 1])
        pos_0 = pos[i, 1:3]
    # 返回值为位置，时间，脚步，步长和对应原始数据的索引
    return pos, steps[:, 0]


#


file = '.\data\\UWB采集数据\\IMU_1.txt'
pos_pdr, index = test_pdr(1, file)

file = '.\data\\UWB采集数据\\UWB_1.txt'
dis = rd.read_file(file)
init_p = np.mat([3.08, 6.22, 0]).T
anchor = np.array([[0, 0, 0], [20.72, 0, 0], [20.72, 11.67, 0], [0, 11.67, 0.1]]).T
# anchor_2=np.mat([[0, 0, 0], [0, 6.22, 0], [20.72, 11.67, 0], [0, 11.67, 0.1]]).T
# ref = np.array([308, 622, 752, 622, 752, 64, 1640, 64,
#                 1640, 982, 308, 982, 308, 622]).reshape(7, 2) / 100
# ref = rd.read_file('pdr_ref.txt')
pos_uwb = np.mat(np.zeros((dis.shape[0], 3)))

# 计算UWB的位置
for i in range(dis.shape[0]):
    init_p = uwb.getPosition(anchor, dis[i, 1:].T)  # uwb.getDe_Position(anchor, init_p, dis[i, 1:].T, 3)
    pos_uwb[i, :] = init_p.T

pos_uwb = imu.del_outlier(4, pos_uwb)

pos_uwb = imu.choose_from(pos_uwb, index)
uwb_ref = pos_uwb.copy()
# np.savetxt('uwb_ref.txt',pos_pdr[:,1:])

pos_fusion = np.zeros(pos_pdr.shape)
P = np.eye(4)
Q = np.diag([1, 1, 1, (15 * D2R) ** 2])
R_1 = np.diag([2 ** 2, 2 ** 2])
R_2 = np.diag([20 ** 2, 20 ** 2])
pos_fusion[0, :] = pos_pdr[0, :]
pre_pos = init_pos
t = pos_fusion[0, 0]
x = [0.0, 0.0]
e = [0.0, 0.0]
yaw_e = 0
# 融合，先做融合修正姿态，然后再做pdr，pdr结束后再做ekf修正位置和姿态
for i in range(0, pos_fusion.shape[0]):
    if i < 30:
        R = R_1
    else:
        R = R_2
    pos_fusion[i,] = imu.pdr(pos_pdr[i, 0], pre_pos, pos_pdr[i, 3] - x[0], pos_pdr[i, 4]-x[1])
    pos_fusion[i, 1:], P, e[0], e[1] = fusion.fusion(pos_uwb[i, :2].T, pos_fusion[i, 1:], P, R, Q)
    x[0] += e[0]
    x[1] += e[1]
    pos_fusion[i, 3] = imu.central_yaw(pos_fusion[i, 3])
    if i >= 30:
        pos_fusion[i, :] = imu.pdr(pos_fusion[i, 0], pos_fusion[i - 1, 1:3], pos_fusion[i, 3], pos_fusion[i, 4])

    pos_fusion[i, 1:], P, e[0], e[1] = fusion.fusion(pos_uwb[i, :2].T, pos_fusion[i, 1:], P, R, Q)
    x[0] += e[0]
    x[1] += e[1]
    pre_pos = pos_fusion[i, 1:3]

# pos_0 = init_pos
# for i in range(pos_fusion.shape[0]):
#     pos_fusion[i, :] = imu.pdr(pos_fusion[i, 0], pos_0, pos_fusion[i, 3], pos_fusion[i, 4])
#     pos_0 = pos_fusion[i, 1:3]

# np.savetxt('./data/UWB_1_pow.txt', pos_pdr)

pow_ref = rd.read_file('./data/UWB_1_pow.txt')
# pos_ref = np.array([308, 622, 308, 64, 752, 64, 752, 622, 1640, 622, 1640, 64,
#                     2084, 64, 2084, 982, 308, 982, 308, 622]).reshape(10, 2) / 100
# pos_ref = np.array([308, 622, 752, 622, 752, 64, 1640, 64,
#                 1640, 982, 308, 982, 308, 622]).reshape(7, 2) / 100
pos_ref = rd.read_file('./data/UWB_1_truth.txt')
fig_2, ax = plt.subplots(figsize=(12, 8))
ax.plot(pos_fusion[:, 0], pos_fusion[:, 3] / D2R, 'rx-.', label='fusion')
ax.plot(pow_ref[:, 0], pow_ref[:, 3] / D2R, 'yx-.', label='ref')
ax.plot(pos_pdr[:, 0], pos_pdr[:, 3] / D2R, 'bx-.', label='pdr')
ax.legend(loc=2)
# np.savetxt('totruth.txt', pos_fusion[:,1:3])
fig_3, ax = plt.subplots(figsize=(12, 8))
# ax.plot(pos_pdr[:, 1], pos_pdr[:, 2], 'bx-.', label='pdr')
ax.plot(pos_uwb[:, 0], pos_uwb[:, 1], 'rx-.', label='uwb')
ax.plot(pos_ref[:, 0], pos_ref[:, 1], 'y', label='ref')
ax.plot(pos_fusion[:, 1], pos_fusion[:, 2], 'gx-.', label='fusion')
ax.legend(loc=2)
ax.set_xlabel('x/m')
ax.set_ylabel('y/m')
ax.set_title('track_show')

# erro_p = pos_ref[:, 0:2] - pos_pdr[:, 1:3]

erro_p = np.zeros(pos_fusion.shape[0])
rms = 0
for i in range(erro_p.shape[0]):
    erro_p[i] = np.sqrt((pos_ref[i, 0] - pos_fusion[i, 1]) ** 2 + (pos_ref[i, 1] - pos_fusion[i, 2]) ** 2)
    rms += erro_p[i] ** 2
# print(np.argmax(erro_p))
# print(pos_fusion[97, :])
print(np.std(erro_p))
print(np.sqrt(np.sum(rms) / len(erro_p)))

# fig_4, ax = plt.subplots(figsize=(12, 8))
# ax.plot(erro_p, 'bx-.', label='error')
# ax.set_xlabel('时间')
# ax.set_ylabel('y/m')
# ax.set_title('error')
plt.show()
