import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

a = 10

R = casadi.SX.sym('R') # Rという名前のsymbolを用意
x_list = [casadi.SX.sym(f'x{i}') for i in range(5)] # list x_0, ..., x_4
y_list = [casadi.SX.sym(f'y{i}') for i in range(5)] # list y_0, ..., y_4
r_list = [R,R,R,2*R,2*R] # r_0, ..., r_4 は R,R,R,2R,2Rで代入.

x_nlp = casadi.vertcat(R, *x_list, *y_list) # 連結した
f_nlp = -R # 評価関数. -Rを最小化する, つまりRを最大化する.
g_nlp = [] # 制約?? いったんは空リストを作ってる?

g_nlp.append(R) # 未治療Rを制約リストに入れる(R>=0を意味する!!!) listに対するappend メソッドを覚えよう

for i in range(5): #各iに対して, 次を制約に加える. rande(5)は0,1,2,3,4の範囲.
    g_nlp.append(x_list[i] - r_list[i]) # x_i - r_i >= 0
    g_nlp.append(a-x_list[i] - r_list[i]) # 中身が0以上という式になるので, これはx_i + r_i <= aということ! 右辺を>= 0にして考える.
    g_nlp.append(y_list[i] - r_list[i])
    g_nlp.append(a-y_list[i] - r_list[i])

for j in range(5): # 各i,jに対して次のeq >= 0 であるという制約を加える!こｒは1+2+3;4 = 10個の制約を課す
    for i in range(0, j):
        eq = (x_list[i] - x_list[j])**2 + (y_list[i]-y_list[j])**2 \
                - (r_list[i] + r_list[j])**2
        g_nlp.append(eq)

g_nlp = casadi.vertcat(*g_nlp) #複数の制約条件を列方向に連結する？*g_nlpはアンパック機能というらしい？
#g_nlpそのものはlist型なので, それをcasadi.vertcatに渡すことは出来ない
#SX型, DM型を引数にとるので, 個別にアンパックしておけば, ひとつひとつはSX型になっていて, casad.vertcatに渡すことができるようになる?

# "x", "f", "g"をキーとする辞書nlpの作成
nlp = {'x': x_nlp,
        'f': f_nlp,
        'g': g_nlp
        } #'x'はキーという. これを使うとr = S()で出した最適解における決定変数値を引っ張ってくることができる.

#solver objectの生成
S = casadi.nlpsol('S', 'ipopt', nlp) #最適化問題nlpを, IPOPTという最適化ライブラリを用いて解く, 解法アルゴリズムの用意をしている.
print(S) #最適解を出力しよう
# 「ipoptで解くので, それをnlpの辞書型で返します. 」

# S は辞書を返す！その辞書の形が, まさしく先程定めたnlpの形なのである


# 最適化を実行（初期値は適当な値で設定）
# 辞書r を定義.
r = S(x0=[1,1,3,8,7,3,1,3,8,3,7], #11個の変数があり, その初期推定値がこれになってる.
      lbg=[0]*31, ubg=[np.inf]*31)  # 制約の下限を0、上限をnp.inf = ∞と設定（31個の制約条件があるので31）
x_opt = np.array(r['x']).ravel()  # 最適化の結果を1次元の配列に変換し, それをx_optという名前で定義する.   nlpで定義したキー'x'を呼び出している. rの返り値はmap型みたいな感じ?なので, 'x'を渡せば最適化結果のx_nlpを返すことができる。
print('x_opt: ', x_opt)  # 文字列と, x_optの出力をしている.

# 結果から半径と座標を抽出
R_opt = x_opt[0]  # 最適な半径.
coord_opt = x_opt[1:].reshape(2,5).T  # 各円の中心座標(x,y)を5行2列の行列として整理

# 結果をプロット
fig, ax = plt.subplots(figsize=(6,6))

# 各円をプロット
for i in range(5):
    radius = R_opt if i <= 2 else 2*R_opt  # 最初の3つは半径R_opt、後の2つは半径が2倍（問題の仮定に従う）
    circle = patches.Circle(xy=coord_opt[i], radius=radius, fill=False)  # 円の形状を作成（塗りつぶしなし）
    ax.add_patch(circle)  # 円を描画
    ax.scatter(*coord_opt[i], marker="x")  # 円の中心をx印でマーク（中心位置確認用）

# 制約となる矩形領域（10×10）の枠を描画
rect = patches.Rectangle([0,0],10,10,fill=False)  # 左下(0,0)を始点に10×10の矩形
ax.add_patch(rect)

# 描画領域の設定（矩形領域に合わせる）
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.axis("equal")  # x軸とy軸の比率を1:1に設定（円が正しく描画されるため）

# 画像として保存
plt.savefig("images/chap3_NLP_packing_1.png")

# 描画を表示
plt.show()
