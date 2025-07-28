import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

x = casadi.SX.sym("x")
y = casadi.SX.sym("y")

qp = {
    "x":casadi.vertcat(x,y),
    "f":x**2 + 2*y**2 - x - 2*y,
    "g":casadi.vertcat( #制約に用いる式
        x+y-3,
        x-1,
        3-y,
        -x+3*y+1,
        5-x-y
    )
    }

S = casadi.qpsol("S", "osqp", qp)
print(S)

r = S(x0=[0,1], lbg=[0,0,0,0,0], ubg=[0,np.inf, np.inf, np.inf, np.inf])
x_opt = np.array(r["x"])
print("x_opt: ", x_opt)

# x+y-3=0, inf >= x-1 >= 0, ... など

#等間隔の数値列の生成. 0から4.00まで, 0.01刻みで並んだ数列がX_
X_, Y_ = np.arange(0,4.01,0.01), np.arange(0,3.51,0.01)
X, Y = np.meshgrid(X_, Y_) #座標軸
Z= X**2 + 2*Y**2 - X - 2*Y #目的関数

levs = np.linspace(-20,30,50) # -20から30までを等間隔に50分割

fig, ax = plt.subplots(figsize=(8,6)) #figが全体, axが一つの描画領域
ax.scatter(x_opt[0], x_opt[1], c="red") #最適解を赤い点で描画
cs = ax.contour(X,Y,Z,levels=levs) #等高線の描画
fig.colorbar(cs) #カラーバーをメモに表示

points = [[1,0], [1,2], [1,3], [1,4], [5/2,1/2]] #制約領域の頂点を手動で与える
polygon = patches.Polygon(xy=points, closed=True, alpha=0.5) #多角形をだす
ax.add_patch(polygon) # グラフに追加

#線分をかくためのパッチで, closed = Flaseによって多角形は直線になる
polyline = patches.Polygon([[1,2], [5/2,1/2]],
                           closed =True, edgecolor="black",
                           facecolor="none", linewidth=2)
ax.add_patch(polyline) #追加

plt.savefig("images/chap3_QP_2D.png")
plt.show()
