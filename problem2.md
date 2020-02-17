# Problem 2
__所有评分在加权之前都进行归一化处理。__
- 战术评分 ALL
  $$\begin{matrix}
    1&1&3&2&2\\
    1&1&2&2&2\\
    1/3&1/2&1&2&1\\
    1/2&1/2&1/2&1&1/2\\
    1/2&1/2&1&1/2&1\\
  \end{matrix}
  $$
  - 进攻 OA
    $$\begin{matrix}
      1&1&2\\
      1&1&2\\
      1/2&1/2&1\\
    \end{matrix}
    $$
    - 进攻者OP
      $$\begin{matrix}
        1&3&1&1&1&1/2\\
        1/3&1&1/3&1/3&1/3&1/7\\
        1&3&1&1&1&1/2\\
        1&3&1&1&1&1/2\\
        1&3&1&1&1&1/3\\
        2&7&2&2&3&1\\
      \end{matrix} $$
      - 控球时间比率
      - simple pass的次数
      - smart pass的次数
      - num1 = ground duel后发生pass的次数
      - -num2 = ground duel后继续为duel的次数
      - num3 = 对方foul后得到本方free kick次数
    - 防守者DP
      - num5 = 对方save attempt次数
    - 特殊情况（角球）flag（发生时，独占权重）
  - 防守 DA
    $$\begin{matrix}
      1&1&1/2\\
      1&1&1/3\\
      2&3&1\\
    \end{matrix}
    $$
    - -ODC
    - -PPDA
    - 对方shot/goal
  - 协调性
    - L2。 拉普拉斯矩阵第二小特征值
  - 鲁棒性 RB=[0.5,0.5]
    - C。聚类系数
    - L1。最大特征值
  - 灵活性
    - 1/d。
