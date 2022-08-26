# Probabilistic Programming

## Introduction

円周率 $\pi$ は Feynman が最も美しい数式[^Feynman]としてあげたオイラーの等式

$$
e^{i\pi} + 1 = 0
$$

にも現れるように、数学上最も重要な概念の一つであるが、無理数であるため数値で表すには無限大の精度が必要である。
そのため計算機上では $\pi$ は厳密な数値としては扱えず、近似的に表す以外ない。
そこでいくつかの方法で円周率を計算してみよう。

一つの方法は次の恒等式

$$
\frac{\pi}{4} = \int^1_0 \frac{dx}{1 + x^2}
$$

で右辺の積分を数値的に実行し、その結果を４倍することで求める。
台形近似で積分を実行

```python
def trapezoid_pi():
    integrand = lambda x: 1.0 / (1.0 + x ** 2.0)
    delta = 1 / N
    x = 0
    area = 0
    for n in range(1, N):
        upper = x + n*delta
        lower = x + (n-1) * delta
        area += 0.5 * delta * (integrand(lower) + integrand(upper))

    return 4.0 * area
```

<p align="center">
  <b>List 1.</b>
  台形近似で円周率を求めるプログラム。
</p>

した結果は $\pi = 3.1411926069191316$ となり、 `numpy`に実装されている円周率との誤差が0.01%である。

| N | pi | error |
| -: | -: | -: |
| 5 | 2.691023675 | 14.342%
| 50 | 3.101121988 | 1.288%
| 500 | 3.137587983 | 0.127%
| 5000 | 3.141192607 | 0.013%
| 50000 | 3.141552653 | 0.001%
| - | 3.141592653 | 0.000%

<p align="center">
  <b>Table 1.</b>
  台形近似を用いた円周率とその誤差。
  最後の行はオンライン整数列大辞典の数列 <a href="https://oeis.org/A000796">A000796</a>の値。
</p>

$$
\frac{\pi}{4} = \int_{D} dx dy,
\quad D: 0 \leq x, y \leq 1,\ x^2 + y^2 \leq 1
$$

この積分を以下のように書き換える:

$$
\frac{\pi}{4}
= \iint^1_0 \theta(1 - x^2 - y^2) dx dy
= \iint^1_0 \frac{p(x,y)\theta(1 - x^2 - y^2)}{p(x,y)} dx dy
\sim \frac{1}{N}\sum_{n=1}^N \frac{\theta(1 - x_n^2 - y_n^2)}{p(x,y)}
= \frac{1}{N}\sum_{n=1}^N \theta(1 - x_n^2 - y_n^2)
$$

ここで $p(x,y)$ は確率密度関数であり区間 $[0,1]$ の一様分布を表す。
$N$ は $p(x,y)$ によってサンプリングされた点の数で、 $x_n, y_n$ はその各点の座標値を表す。
また $theta(x)$ はステップ関数である。

この方法では積分を実行するのに一様分布からサンプリングされた点を用いている。
このように何らかの確率から生成された乱数を用いて計算を行うことを**モンテカルロ法**、特に積分に用いたものを**モンテカルロ積分**という。
積分を実行するプログラムをList 2.に示す。
またその結果をTable 2.に示す。

```python
def calc_pi():

    for i in tqdm(range(N)):
        x, y = np.random.rand(2)
        if x ** 2 + y ** 2 <= 1.0:
            accepted[0].append(x)
            accepted[1].append(y)
            # plt.plot(x, y, marker="o", c="g", alpha=0.3, mec="g")
        else:
            rejected[0].append(x)
            rejected[1].append(y)
            # plt.plot(x, y, marker="o", c="r", alpha=0.3, mec="r")

    pi = 4.0 * len(accepted[0]) / N
    return pi
```

<p align="center">
<b>List 2.</b>
モンテカルロ法で円周率を求めるプログラム
</p>

| N | pi | error |
| -: | -: | -: |
| 5 | 4.0 | 27.324%
| 50 | 3.04 | 3.234%
| 500 | 3.112 | 0.942%
| 5000 | 3.1488 | 0.229%
| 50000 | 3.14376 | 0.069%
| - | 3.14159265 | 0.000%

<p align="center">
  <b>Table 2.</b>
  モンテカルロ法を用いた円周率とその誤差
</p>

<p align="center">
  <img width="460" height="auto" src="./fig/monte_carlo_pi.png">
</p>

<p align="center">
  <b>Figure 1.</b>
  Monte Carlo pi
</p>

## Importance sampling

## Monte Carlo Markov Chain method

マルコフ連鎖[^2018Fukushima]

### Detailed balance condition

熱平衡にある２つの状態 $|i\rangle, |j\rangle$ が互いにある確率で遷移するとする。
状態 $|i\rangle$ から状態 $|j\rangle$ への遷移確率を

$$
p(|i\rangle \to |j\rangle) \equiv p(j|i) \equiv p_{i,j},
$$



### Metropolis-Hasting algorithm

### Heat bath algolithm

### Hamiltonian Monte Carlo

## Self-Learning Monte Carlo[^2019Nagai]

[^Feynman]: 要出典

[^2018Fukushima]: 福島孝治, < 講義ノート> モンテカルロ法の基礎と応用--計算物理学からデータ駆動科学へ--, 物性研究・電子版, 2018, 7.2: 1-10.,
https://doi.org/10.14989/235551

[^2019Nagai]: 永井 佑紀, 自己学習モンテカルロ法：機械学習を用いたマルコフ連鎖モンテカルロ法の加速, アンサンブル, 2019, 21 巻, 1 号, p. 15-21, 公開日 2020/01/31, Online ISSN 1884-5088, Print ISSN 1884-6750, https://doi.org/10.11436/mssj.21.15, https://www.jstage.jst.go.jp/article/mssj/21/1/21_15/_article/-char/ja, 抄録:

