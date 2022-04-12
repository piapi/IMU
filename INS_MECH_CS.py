import numpy as np
import pandas as pd
import INS_MECH_CLASS as cla
import INS_MECH_FUNCTION as fnc
import os
import struct
import matplotlib.pyplot as plt

import file_test.readfile as rd


def main():
    meas_cur = np.array([[0.0], [0], [0], [0], [0], [0], [0]])
    meas_prev = np.array([[91620.005000000], [0], [0], [0], [0], [0], [0]])
    GPS = np.array([[0.0], [0], [0], [0], [0], [0], [0]])
    Cbn = fnc.euler2dcm(np.pi * 0.010832866167475651 / 180, np.pi * -2.1424872147973986 / 180,
                        np.pi * -75.74984266985793 / 180)

    nav = cla.Nav(
        r=np.array([[np.pi * 23.13739500000392 / 180], [np.pi * 113.37136499999202 / 180], [2.1749993762671918]]), C_bn=Cbn,
        v=np.array([[0.00017423243563459336], [-0.0003269947225345985], [0.0002494931229694991]]))

    nav.q_bn = fnc.dcm2quat(Cbn)
    nav.q_ne = fnc.pos2quat(nav.r[0, 0], nav.r[1, 0])
    par = cla.Par()

    temp = [91620.005000000,
            nav.r[0, 0] * 180 / np.pi, nav.r[1, 0] * 180 / np.pi, nav.r[2, 0],
            nav.v[0, 0], nav.v[1, 0], nav.v[2, 0],
            (fnc.dcm2euler(nav.C_bn).T * 180 / np.pi)[0, 0],
            (fnc.dcm2euler(nav.C_bn).T * 180 / np.pi)[0, 1],
            (fnc.dcm2euler(nav.C_bn).T * 180 / np.pi)[0, 2]
            ]
    # 参考
    times = 720000
    #
    # bfile = open('Reference.bin', 'rb')  # 打开二进制文件
    # size = os.path.getsize('Reference.bin')  # 获得文件大小
    # f = open("ref_ins.txt", "a")  # 利用追加模式,参数从w替换为a即可
    # f.truncate(0)
    #
    # index = 0
    # for i in range(size):
    #     data = bfile.read(8)  # 每次输出8个字节
    #     num = struct.unpack('d', data)
    #     GPS[i % 10, 0] = num[0]
    #     if (i + 1) % 10 == 0:
    #         res_ref = [GPS[0, 0], GPS[1, 0], GPS[2, 0], GPS[3, 0], GPS[4, 0],
    #                 GPS[5, 0], GPS[6, 0], GPS[7, 0], GPS[8, 0], GPS[9, 0]]
    #         f.write("{}\n".format(res_ref))
    #         index += 1
    #         if index == times:
    #             break
    # f.close()
    # statTime = 91620.005
    # f = open("./data/INS.txt", "a")  # 利用追加模式,参数从w替换为a即可
    # f.truncate(0)
    # f.write("{}\n".format(temp))
    # binfile = open('./data/IMU.bin', 'rb')  # 打开二进制文件
    # size = os.path.getsize('./data/IMU.bin')  # 获得文件大小
    # index = 0
    # for i in range(size):
    #     data = binfile.read(8)  # 每次输出8个字节
    #     num = struct.unpack('d', data)
    #     meas_cur[i % 7, 0] = num[0]
    #     if (i + 1) % 7 == 0 and meas_cur[0, 0] <= statTime:
    #         meas_prev[:] = meas_cur[:]
    #     elif (i + 1) % 7 == 0 and meas_cur[0, 0] > statTime:
    #         nav1 = fnc.INS_MECH_CS(meas_prev, meas_cur, nav)
    #         # nav = feed.INS_MECH_BACK(meas_prev, meas_cur, nav1, par)
    #         # nav1 = INS_MECH_CS(meas_prev, meas_cur, nav, par)
    #
    #         meas_prev[:] = meas_cur[:]
    #         nav = nav1.copy()
    #         temp = [meas_cur[0, 0],
    #                 nav.r[0, 0] * 180 / np.pi, nav.r[1, 0] * 180 / np.pi, nav.r[2, 0],
    #                 nav.v[0, 0], nav.v[1, 0], nav.v[2, 0],
    #                 (fnc.dcm2euler(nav.C_bn).T * 180 / np.pi)[0, 0],
    #                 (fnc.dcm2euler(nav.C_bn).T * 180 / np.pi)[0, 1],
    #                 (fnc.dcm2euler(nav.C_bn).T * 180 / np.pi)[0, 2]
    #                 ]
    #
    #         f.write("{}\n".format(temp))
    #         if times / (index + 1) == 5:
    #             print('20%')
    #         elif times / (index + 1) == 2:
    #             print('50%')
    #         elif times / (index + 1.0) == 1.25:
    #             print('80%')
    #         index += 1
    #         if index == times:
    #             break
    # print(index)
    # f.close()

    ylabel = ['lat/deg', 'lot/deg', 'altitude/m', 'Vx/m/s', 'Vy/m/s', 'Vz/m/s', 'roll/deg', 'pitch/deg', 'heading/deg']
    title = ['lat_compare', 'lot_compare', 'altitude_compare', 'Vx_compare', 'Vy_compare', 'Vz_compare', 'roll_compare',
             'pitch_compare', 'yaw_compare']

    res_imu = rd.read_file('./data/INS.txt', 1)
    print(res_imu.shape)
    res_ref = rd.read_file('./data/ref_ins.txt', 1)
    print(res_ref.shape)

    # def DR(RM, RN, h, lat):  # NED2BLH
    #     return np.diag([RM + h, (RN + h), -1])
    #
    # a = 6378137.0
    # e2 = 0.0066943799901413156
    # for i in range(res_imu.shape[0]):
    #     rm = fnc.GetRM(a, e2, res_imu[i, 1])
    #     rn = fnc.GetRN(a, e2, res_imu[i, 1])
    #     res_imu[i, 1:4] = ((DR(rm, rn, res_imu[i, 3], res_imu[i, 1])) @ res_imu[i, 1:4].T).T
    # for i in range(res_ref.shape[0]):
    #     rm = fnc.GetRM(a, e2, res_ref[i, 1])
    #     rn = fnc.GetRN(a, e2, res_ref[i, 1])
    #     res_ref[i, 1:4] = ((DR(rm, rn, res_ref[i, 3], res_ref[i, 1])) @ res_ref[i, 1:4].T).T

    for i in range(9):
        fig_i, ax = plt.subplots(figsize=(12, 8))

        # ax.plot(res_imu[:, 0], res_imu[:, i + 1], 'r', label='imu')
        # ax.plot(res_ref[:, 0], res_ref[:, i + 1], 'b', label='ref')
        # ax.plot(res_ref[:, 1], res_ref[:, 2], 'r', label='guiji')
        ax.plot(res_imu[0:times, 0], res_imu[0:times, i + 1] - res_ref[0:times, i + 1], 'y', label='Compare')

        ax.legend(loc=2)
        ax.set_xlabel('time/s')
        ax.set_ylabel(ylabel[i])
        ax.set_title(title[i])
    plt.show()


if __name__ == '__main__':
    main()
