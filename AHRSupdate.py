import numpy as np

# sampleFreq = 200
Ki = 0.05
Kp = 3.5


def MahonyAHRSupdateIMU(ax, ay, az, gx, gy, gz, q, sample_rate):
    halfT = 2 / sample_rate
    if not (ax == 0.0 and ay == 0.0 and az == 0):
        # Normalise acceleromete measurement
        recipNorm = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        ax = ax / recipNorm
        ay = ay / recipNorm
        az = az / recipNorm
        # Estimated direction of gravity and vector perpendicular to magnetic flux
        vx = 2 * (q[1] * q[3] - q[0] * q[2])
        vy = 2*(q[0] * q[1] + q[2] * q[3])
        vz = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
        # Error is sum of cross product between estimated and measured direction of gravity
        ex = ay * vz - az * vy
        ey = az * vx - ax * vz
        ez = ax * vy - ay * vx
        exInt = 0.0
        eyInt = 0.0
        ezInt = 0.0
        exInt += ex * Ki
        eyInt += ey * Ki
        eyInt += ez * Ki

        gx += Kp * ex + exInt
        gy += Kp * ey + eyInt
        gz += Kp * ez + ezInt

        qa = q[0]
        qb = q[1]
        qc = q[2]
        q[0] += -qb * gx - qc * gy - q[3] * gz
        q[1] += qa * gx + qc * gz - q[3] * gy
        q[2] += qa * gy - qb * gz + q[3] * gx
        q[3] += qa * gz + qb * gy - qc * gx

        q = q * halfT
        recipNorm = np.linalg.norm(q)
        q = q / recipNorm

    return q
