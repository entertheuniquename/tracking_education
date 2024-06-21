#!/usr/bin/python3

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
import matplotlib.pyplot as plt
import math

# Функции для EKF (см. предыдущие шаги)
def fx(x, dt):
    theta = x[3]
    omega = x[4]
    
    if abs(omega) > 1e-4:
        new_x = x[0] + (x[2] / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        new_y = x[1] + (x[2] / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
    else:
        new_x = x[0] + x[2] * dt * np.cos(theta)
        new_y = x[1] + x[2] * dt * np.sin(theta)

    new_theta = theta + omega * dt

    return np.array([new_x, new_y, x[2], new_theta, omega])

def Fx(x, dt):
    theta = x[3]
    omega = x[4]

    if abs(omega) > 1e-4:
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_omega_dt = np.sin(theta + omega * dt)
        cos_theta_omega_dt = np.cos(theta + omega * dt)
        
        V_over_omega = x[2] / omega
        dfdx = np.array([
            [1, 0, (sin_theta_omega_dt - sin_theta) / omega, 
                V_over_omega * (cos_theta_omega_dt - cos_theta), 
                x[2] * (cos_theta_omega_dt * dt - (sin_theta_omega_dt - sin_theta) / omega) / omega],
            [0, 1, (cos_theta - cos_theta_omega_dt) / omega, 
                V_over_omega * (sin_theta - sin_theta_omega_dt), 
                x[2] * (sin_theta_omega_dt * dt - (cos_theta - cos_theta_omega_dt) / omega) / omega],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, dt],
            [0, 0, 0, 0, 1]
        ])
    else:
        dfdx = np.array([
            [1, 0, dt * np.cos(theta), -x[2] * dt * np.sin(theta), 0],
            [0, 1, dt * np.sin(theta), x[2] * dt * np.cos(theta), 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, dt],
            [0, 0, 0, 0, 1]
        ])
        
    return dfdx

def hx(x):
    return np.array([x[0], x[1]])

def Hx(x):
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ])

# Инициализация EKF
ekf = EKF(dim_x=5, dim_z=2)

# Начальное состояние
ekf.x = np.array([10, 0, 1, np.pi / 2, 0.0])  # [x_position, y_position, velocity, heading, turn_rate]

# Ковариационная матрица начального состояния
ekf.P = np.eye(5) * 500.

# Создание ковариационной матрицы шума процесса вручную
q = 0.1
ekf.Q = np.diag([q, q, q, q, q])

# Ковариационная матрица измерений
ekf.R = np.eye(2) * 2

# Параметры окружности для измерений
R = 10  # радиус
num_points = 10  # количество точек
dt = 6.0  # шаг времени

# Генерация углов
angles = np.linspace(0, 2 * np.pi, num_points)
print("angles:")
print(angles)

# Генерация точек на окружности
measurements = np.array([[R * np.cos(angle), R * np.sin(angle)] for angle in angles])
print("measurements:")
print(measurements)
measurements = np.vstack([measurements,measurements])
measurements = np.vstack([measurements,measurements])
print("measurements:["+str(measurements.shape[0])+","+str(measurements.shape[1])+"]")
print(measurements)

a = np.random.normal(loc=0, scale=math.sqrt(1.0), size=(measurements.shape[0], measurements.shape[1]))
print("a["+str(a.shape[0])+","+str(a.shape[1])+"]:")
print(a)

at = a.transpose()
print("at["+str(at.shape[0])+","+str(at.shape[1])+"]:")
print(at)

r = np.sqrt(ekf.R)
print("r["+str(r.shape[0])+","+str(r.shape[1])+"]:")
print(r)

rat = r @ at
print("rat["+str(rat.shape[0])+","+str(rat.shape[1])+"]:")
print(rat)

m = measurements.transpose()
print("m["+str(m.shape[0])+","+str(m.shape[1])+"]:")
print(m)

m_res = m + rat
print("m_res["+str(m_res.shape[0])+","+str(m_res.shape[1])+"]:")
print(m_res)

measurements_noise = m_res.transpose()

#measurements = measurements + np.sqrt(ekf.R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(measurements.shape[0], measurements.shape[1]))
# Списки для хранения результатов
ekf_positions = []
estimated_omegas = []

# Процесс предсказания и обновления
for z in measurements_noise:
    ekf.F = Fx(ekf.x, dt)  # матрица перехода состояний
    ekf.predict()
    ekf.update(z, HJacobian=Hx, Hx=hx)
    ekf_positions.append(ekf.x[:2])
    estimated_omegas.append(ekf.x[4])

ekf_positions = np.array(ekf_positions)

# Визуализация результатов
plt.figure(figsize=(24, 10))

plt.subplot(1, 2, 1)
plt.plot(measurements[:, 0], measurements[:, 1], '-', label='Measurements', color='orange')
plt.plot(measurements_noise[:, 0], measurements_noise[:, 1], '+', label='Measurements', color='red')
plt.plot(ekf_positions[:, 0], ekf_positions[:, 1], '-', label='EKF', color='purple')
plt.xlabel('x')
plt.ylabel('y')
plt.title('EKF with Circular Measurements')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(3, 2, 2)
plt.plot(estimated_omegas, '-', label='Estimated Omega', color='purple')
plt.xlabel('Step')
plt.ylabel('Omega')
plt.title('Estimated Omega over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(measurements[:, 0], '-', label='measurements-x', color='orange')
plt.plot(measurements_noise[:, 0], '+', label='measurements-x', color='red')
plt.plot(ekf_positions[:, 0], '-', label='EKF-x', color='purple')
plt.xlabel('Step')
plt.ylabel('X')
plt.title('Estimated X over Time')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(measurements[:, 1], '-', label='measurements-x', color='orange')
plt.plot(measurements_noise[:, 1], '+', label='measurements-y', color='red')
plt.plot(ekf_positions[:, 1], '-', label='EKF-y', color='purple')
plt.xlabel('Step')
plt.ylabel('Y')
plt.title('Estimated Y over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
