import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

x = casadi.SX.sym("x")
y = casadi.SX.sym("y")


#係数
a = 0.06
b = 0.06
c = 0.024
d = 0.06
e = 0.06
f = 0.012

#ホライズン
K = 30

nx = 2 #状態変数の次元
nu = 1 #制御入力の次元
total = nx*(K+1) + nu*K #決定変数の次元(2.2節にあるのと同じ話)

#評価関数の重み
Q = casadi.diag([1,1]) #対角行列
Q_f = casadi.diag([1,1])
R = casadi.diag([0.05,0.05])

#制約条件
x_lb = [0,0]
x_ub = [np.inf, np.inf]
u_lb = [0] #入力下限は0
u_ub = [1] #入力上限は1

# 目標値
x_ref = casadi.DM([1,1])
u_ref = casadi.DM([0])

#4.6.2
def make_F():
    states = casadi.SX.sym("states", nx)
    ctrls = casadi.SX.sym("states",nu)

    x = states[0]
    y = states[1]
    u = ctrls[0]

#状態方程式モデル
    x_next = (1+a)*x - b*x*y - c*x*u
    y_next = (1-d)*x + e*x*y - f*y*u

    states_next = casadi.vertcat(x_next,y_next) #次の状態空間の点

    F = casadi.Function("F", [states,ctrls], [states_next])
    return F

# 4.6.3 制御入力0 = 漁を行わない場合の個体数増減のサイクル現象観察

t_span = [0,200]
t_eval = np.arange(*t_span)
x_init = casadi.DM([0.5,0.7]) #初期値

F = make_F()

X = [x_init]
x_current = x_init
for t in t_eval:
    x_current = F(x=x_current)["x_next"]  #?
    X.append(x_current)

X.pop() #どこが消えた 最後に追加したやつだけ消えたのか？
X = np.array(X).reshape(t_eval.size,nx)

plt.figure(figsize=(12,4))

plt.subplot(1,1,1)
for k in range(nx):
    plt.plot(t_eval,X[:,k],label=f"x_{k}")
plt.legend()

plt.savefig("images/chap4_mpc_no_contorl.png")
plt.show()
