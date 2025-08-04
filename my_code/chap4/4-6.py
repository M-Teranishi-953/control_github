import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

x = casadi.SX.sym("x")
y = casadi.SX.sym("y")


#==============note=================
# 生物個体数管理の制御問題に対して,
# ロトカ・ヴォルテラのモデルの状態方程式というのを実装し,
# MPCで最適入力を求める.


# 4.6.1 定数の決定
#     漸化式の定数部分や、ホライズンや、決定変数の次元、制御入力の次元、評価関数の重みを定義する.
# 4.6.2 状態方程式の作成.
#     Fを作る. 微分方程式の定義.
# 4.6.3 個体数増減の様子
#     作ったFで、増減の様子を図示するということをやる
# 4.6.4 評価関数の決定
#     被食者と捕食者の数を安定化させるために, 適切な制御入力を求めたい. 最適化問題の作成をし, 評価関数を定義する.
# 4.6.5 最適化問題の定式化
#     casadiを用いて式4.6の定義をする.
# 4.6.6 最適な制御入力を出力する関数の決定
#     最適な制御入力を出力する関数compute_optimal_controlを定義する.
# 4.6.7 MPCの実行

# 4.6.8 結果の可視化
#     表示するだけ.
#==============note=================



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

#評価関数の重みをつける二次pos definite matrix
Q = casadi.diag([1,1]) #状態側の行列. staging cost用
Q_f = casadi.diag([1,1]) #terminal costに使うやつ
R = casadi.diag([0.05,0.05]) # 入力側の行列. staging cost用

#制約条件
x_lb = [0,0]
x_ub = [np.inf, np.inf]
u_lb = [0] #入力下限は0
u_ub = [1] #入力上限は1

# 目標値
x_ref = casadi.DM([1,1])
u_ref = casadi.DM([0])

#4.6.2 ロトカ・ヴォルテラ方程式の漁制御問題
# x_{k+1} = F(x_k, u_k) を表現したい.
# make_F は, casadi.Functionオブジェクトを作成する関数.
def make_F():
    states = casadi.SX.sym("states", nx)
    ctrls = casadi.SX.sym("states",nu)

    x = states[0] #被食者の割合
    y = states[1] #捕食者の割合
    u = ctrls[0]

#状態方程式モデルの記述 (4.1)
#x,yについて線型だったり双線型だったりするとみなして, そのもとでモデルを組む.
    x_next = (1+a)*x - b*x*y - c*x*u # 次の時間での被食者の量 = 自然な増加 - 捕食された被食者 - 人間の漁の量
    y_next = (1-d)*x + e*x*y - f*y*u # 次の時間での捕食者の量 = 自然な減少 + 捕食したことによる捕食者増加 - 人間の漁の漁

    states_next = casadi.vertcat(x_next,y_next) #次の状態空間の点をstates_nextとおく(R^2の点).

    F = casadi.Function("F", [states,ctrls], [states_next],
                        ["x","u"], ["x_next"])
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


#4.6.4 評価関数の決定.
# section 4.3.
# 個体数の変動を抑えて一定にすることが望ましい.
# 状態の目標値をx_refとし、入力の目標値をu_refとして、xとx_refの差、uとu_refの差を記述するような関数が望ましいであろう.
# ステージコストと終端コストがあり,
# stage cost :  L(x,u)=1/2 (x-x_ref)^T Q(x-x_ref) + 1/2 (u-u_ref)^T R (u-u_ref) とのように, 二次関数的に考える.
# terminal cost : \phi(x) = 1/2(x-x_ref)^T Q_f(x-x_ref) とのように、入力はないので状態についてだけ二次関数的なコストを与える.

# compute_stage_costの定義
def compute_stage_cost(x,u):
    x_diff = x - x_ref
    u_diff = u - u_ref
    cost = (casadi.dot(Q@x_diff,x_diff) #
            + casadi.dot(R@u_diff,u_diff)) / 2
    return cost

def compute_terminal_cost(x):
    x_diff = x - x_ref
    cost = casadi.dot(Q_f@x_diff,x_diff) / 2
    return cost

# 4.6.5 最適化問題の定式化
def make_nlp():
    F = make_F() # Function objectの作成

    X = [casadi.SX.sym(f"x_{k}", nx) for k in range(K+1)] # k=0,1,...,K に対して, x_{ks + k} を表すSX object listを作成
    U = [casadi.SX.sym(f"u_{k}", nu) for k in range(K)] #k=0,1,...,K-1 に対して, u_{ks + k} を表すSX object listを作成/
    G = [] #空リスト. ここに制約条件がappendで格納されていく.

    J = 0 #評価関数の値になる.

    for k in range(K):
        J+=compute_stage_cost(X[k],U[k]) # Stage Cost! L(x_k,u_k)の値を計算し, Jに加算していく. compute_stage_costは4.6.4にて.
        eq = X[k+1] - F(x=X[k],u=U[k])["x_next"] #等式条件のためのeqを定義
        G.append(eq) #制約に追加(後にub, lb = 0にする)
    J += compute_terminal_cost(X[-1]) #terminal cost!

    #辞書の定義
    option = {"print_time":False, "ipopt":{"print_level" :0}} #solverのoptionをまとめた辞書. 出力を抑える目的でFlase, 0にしてるらしい.
    nlp = {
        "x":casadi.vertcat(*X,*U), #決定変数
        "f":J, #評価関数
        "g":casadi.vertcat(*G) #制約. (unpacked)
    }
    S = casadi.nlpsol("S","ipopt",nlp,option)
    return S ###

#4.6.6 最適な制御入力を出力する関数の決定
def compute_optimal_control(S,x_init,x0):
    x_init = x_init.full().ravel.tolist() # x_initをnumpy.ndarrayに変換し, 1次元配列に変換し, Pythonのリストにするという操作らしい.
#決定変数は x_{ks+0}, ..., x_{ks+K}, u_{ks+0}, ..., u_{ks+K-1}と並んでいる. これに注意してリストを作る.
    lbx = x_init + x_lb*K + u_lb*K #この+はリストの連結！状態と入力は0以上であることを制約にする.
    ubx = x_init + x_ub*K + u_ub*K #これも連結！状態は上限なし(infty), 入力は1以下にする！
    lbg = [0]*nx*K #制約は全部等式だから, lbgもubgも全部0.
    ubg = [0]*nx*K

    res = S(lbx=lbx, ubx=ubx, lbg=lbg, ung=ubg, x0=x0) #solver適用. 辞書nlpの通りに出てくる.

    offset = nx*(K+1) #取り出したいu0は何番目にいるのかという部分.
    x0 = res["x"] #決定変数をキーで取り出した.
    u_opt = x0[offset:offset+nu] #得られた制御入力u_0. これはスライス記法である. 半開区間[offset, offset + nu) をイメージする.,
    return u_opt, x0 #compute_optimal_controlは, 最適入力とx0を渡す.


#4.6.7 MPCの実行
S = make_nlp()

t_span = [0,200]
t_eval = np.arange(*t_span,1)

x_init = casadi.DM([0.5,0.7])
x0 = casadi.DM.zeros(total)

F = make_F()

X = [x_init]
U = []
x_current = x_init
for t in t_eval:
    u_opt,x0 = compute_optimal_control(S,x_current,x0)
    x_current = F(x=x_current, u=u_opt)["x_next"]
    X.append(x_current)
    U.append(u_opt)


#4.6.8 結果の可視化
X.pop()
X = np.array(X).reshape(t_eval.size,nx)
U = np.array(U).reshape(t_eval.size,nu)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
for k in range(nx):
    plt.plot(t_eval,X[:,k],label=f"x_{k}")
plt.legend()

plt.subplot(1,2,2)
for k in range(nu):
    plt.step(t_eval,U[:,k],linestyle="--",label=f"u_{k}")
plt.legend

plt.savefig("images/chap4_mpc.png")
plt.show()
