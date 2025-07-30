import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

dt = 0.1
t_eval = np.arange(0,10+dt, dt) # 0から10を0.1刻みにする. これはnumpy配列.
X = [casadi.DM([1,1])]

#ODEに現れる行列を定義
A = casadi.DM([
    [0,1],
    [-10,-2]
])
x = casadi.SX.sym("x", 2) #2次元ベクトル

#Aとxをかける
ode = A@x #線型だからこう書いちゃえばいいのか
dae = {"x":x, "ode":ode} #key xで考えるべきxを取る. key ode で微分方程式をだす. 辞書.

F = casadi.integrator("F", "idas", dae,0,dt) #数値解の計算.

for t in t_eval[:-1]: #最初から、最後の一つ前まで！終端は使わない。インデックスで-1を使うとき、末尾から数えることに対応する.
    res = F(x0=X[-1]) #X[-1]は現在時刻の状態ベクトX(t)を表すそう。
    X.append(res["xf"])  #Xにresが格納されていく. "xf"はキーであり, 終端値を返す.

print(res.keys()) # 出力: dict_keys(['adj_p', 'adj_u', 'adj_x0', 'adj_z0', 'qf', 'xf', 'zf'])

X = np.array(X).reshape(-1,2)

#ODEの理論から数学的に解いた解析解. t_evalを変数っぽく書いて、関数に対してリストが作れる. 
X_true_1 = casadi.exp(-t_eval)*casadi.cos(3*t_eval) \
            + 2/3*casadi.exp(-t_eval)*casadi.sin(3*t_eval)
X_true_2 = casadi.exp(-t_eval)*casadi.cos(3*t_eval) \
            - 11/3*casadi.exp(-t_eval)*casadi.sin(3*t_eval)


plt.plot(t_eval, X[:,0], #X[:,0]は状態ベクトルxの一列目. Xのすべての行について, 0列目の値を取り出す. (左だけとる)
         label = "数値解_x1", color="blue")
plt.plot(t_eval, X[:,1], #X[]:,1]は2列目 Xのすべての行について, 1列目を取り出す.
         label = "数値解_x2", color="purple")
plt.plot(t_eval, X_true_1,
         label = "解析解_x1", linestyle="--")
plt.plot(t_eval, X_true_1,
         label = "解析解_x2", linestyle="--")

plt.legend()
plt.savefig("images/chap3_integ.png")
plt.show()
