import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

dt = 0.1
t_eval = np.arange(0,10+dt, dt)
X = [casadi.DM([1,1])]

A = casadi.DM([
    [0,1],
    [-10,-2]
])
x = casadi.SX.sym("x", 2)

ode = A@x #線型だからこう書いちゃえばいいのか
dae = {"x":x, "ode":ode}

F = casadi.integrator("F", "idas", dae,0,dt)

for t in t_eval[:-1]:
    res = F(x0=X[-1])
    X.append(res["xf"])

X = np.array(X).reshape(-1,2)

X_true_1 = casadi.exp(-t_eval)*casadi.cos(3*t_eval) \
            + 2/3*casadi.exp(-t_eval)*casadi.sin(3*t_eval)
X_true_2 = casadi.exp(-t_eval)*casadi.cos(3*t_eval) \
            - 11/3*casadi.exp(-t_eval)*casadi.sin(3*t_eval)

plt.plot(t_eval, X[:,0],
         label = "数値解_x1", color="blue")
plt.plot(t_eval, X[:,1],
         label = "数値解_x2", color="purple")
plt.plot(t_eval, X_true_1,
         label = "解析解_x1", linestyle="--")
plt.plot(t_eval, X_true_1,
         label = "解析解_x2", linestyle="--")

plt.legend()
plt.savefig("images/chap3_integ.png")
plt.show()
