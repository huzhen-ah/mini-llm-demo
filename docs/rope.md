# RoPE 原理与实现

本文记录项目中 RoPE 的数学直觉和实现细节。Keras / TensorFlow 与 PyTorch 两个版本采用相同的复数旋转方案、位置约定和张量布局，因此共用本文。

对应代码位于：

| 模块 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| RoPE | `keras-mini-llm/rope.py` | `pytorch-mini-llm/rope.py` |
| Attention 调用 | `keras-mini-llm/attention.py` | `pytorch-mini-llm/attention.py` |

本文会先介绍传统绝对位置编码 APE，再介绍 RoPE，最后说明两者在 attention 中的主要区别，以及本项目 `rope_exp()` 的实现方式。

## 1. 为什么需要位置编码

Transformer 的 self-attention 主要根据 token 向量之间的相似度计算注意力。只看 token embedding 本身，模型并不知道某个 token 出现在第几个位置。

因此需要额外注入位置信息。

注入位置信息有很多种方式，例如：

- 把位置向量加到 token embedding 上。
- 把位置向量和 token embedding 拼接。
- 在 attention score 中加入 position bias。
- 对 query / key 做位置相关变换，例如 RoPE。

经典 Transformer 的 sinusoidal absolute positional encoding 使用的是第一种方式：

```text
h_pos = x_pos + PE(pos)
```

其中：

```text
x_pos   : 第 pos 个 token 的语义向量
PE(pos) : 第 pos 个位置的位置向量
h_pos   : 注入位置信息后的 token 表示
```

## 2. APE：绝对位置编码

### 2.1 sinusoidal APE 的定义

设模型维度为 `d`，每两个维度作为一组。对第 `i` 组，定义频率：

```text
theta_i = 10000^(-2i / d)
```

经典公式通常写成：

```text
PE(pos, 2i)     = sin(pos * theta_i)
PE(pos, 2i + 1) = cos(pos * theta_i)
```

为了理解旋转，使用标准二维坐标顺序 `[cos, sin]` 会更自然：

```text
p_i(pos) = [
  cos(pos * theta_i),
  sin(pos * theta_i)
]
```

这里固定的是同一个频率维度 `i`。完整的 `PE(pos)` 是很多个二维频率块拼起来：

```text
PE(pos) = [
  p_0(pos),
  p_1(pos),
  ...,
  p_{d/2-1}(pos)
]
```

所以 sinusoidal APE 本质上是：

```text
用多组不同频率的周期函数编码同一个位置 pos。
```

### 2.2 为什么 sin/cos 要成对

如果只用 `sin`，会有歧义：

```text
sin(a) = sin(pi - a)
```

只知道 `sin(a)`，相当于只知道单位圆上的高度，不知道点在圆的左边还是右边。

而成对使用：

```text
[cos(a), sin(a)]
```

就相当于知道单位圆上的完整坐标。它表达的是当前位置在某个频率圆上的完整相位。

### 2.3 APE 中的旋转关系

固定某个频率 `theta_i`：

```text
p_i(pos) = [
  cos(pos * theta_i),
  sin(pos * theta_i)
]
```

位置增加 `k` 后：

```text
p_i(pos + k) = [
  cos((pos + k) * theta_i),
  sin((pos + k) * theta_i)
]
```

令：

```text
a = pos * theta_i
b = k * theta_i
```

根据三角恒等式：

```text
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
```

可得：

```text
p_i(pos + k) = R_i(k) p_i(pos)
```

其中：

```text
R_i(k) =
[
  [cos(k * theta_i), -sin(k * theta_i)],
  [sin(k * theta_i),  cos(k * theta_i)]
]
```

也就是说，在同一个频率块中，位置从 `pos` 变成 `pos + k`，等价于在二维平面中逆时针旋转 `k * theta_i`。

完整的 `PE(pos)` 是多个频率块拼接，因此整体上可以写成块对角形式：

```text
PE(pos + k) = R(k) PE(pos)
```

其中：

```text
R(k) = diag(R_0(k), R_1(k), ..., R_{d/2-1}(k))
```

### 2.4 APE 如何参与 attention

APE 的输入形式是：

```text
h_m = x_m + p_m
h_n = x_n + p_n
```

然后计算 query / key：

```text
q_m = W_q h_m
k_n = W_k h_n
```

代入：

```text
q_m = W_q(x_m + p_m)
k_n = W_k(x_n + p_n)
```

attention score 为：

```text
score(m, n) = q_m^T k_n
```

展开：

```text
score(m, n)
= (W_q x_m + W_q p_m)^T (W_k x_n + W_k p_n)
```

令：

```text
A = W_q^T W_k
```

得到：

```text
score(m, n)
= x_m^T A x_n
+ x_m^T A p_n
+ p_m^T A x_n
+ p_m^T A p_n
```

这四项分别对应：

```text
语义-语义
语义-位置
位置-语义
位置-位置
```

需要注意：APE 也可以在位置-位置项中产生相对位置信息。

如果忽略 `A`，只看：

```text
p_m^T p_n
```

那么：

```text
p_m^T p_n = cos((m - n) * theta_i)
```

这确实只和相对距离 `m - n` 有关。

但在真实 attention 中一般是：

```text
p_m^T A p_n
```

并且还有：

```text
x_m^T A p_n
p_m^T A x_n
```

这些项会把 token 语义和绝对位置混合在一起。因此 APE 不是不能表达相对位置，而是它把位置作为额外特征加进 token 表示中，后续由模型自己学习如何使用这些混合特征。

## 3. RoPE：旋转位置编码

### 3.1 从复数乘法理解旋转

设二维向量：

```text
v = [x, y]
```

对应复数：

```text
z = x + iy
```

欧拉公式：

```text
e^{iθ} = cosθ + i sinθ
```

复数乘法：

```text
z' = z * e^{iθ}
```

展开：

```text
z'
= (x + iy)(cosθ + i sinθ)
= (xcosθ - ysinθ) + i(xsinθ + ycosθ)
```

对应到二维向量：

```text
[
  x'
  y'
]
=
[
  [cosθ, -sinθ],
  [sinθ,  cosθ]
]
[
  x
  y
]
```

所以，在复数域中乘以 `e^{iθ}`，等价于在二维实向量空间中左乘旋转矩阵：

```text
R(θ) =
[
  [cosθ, -sinθ],
  [sinθ,  cosθ]
]
```

注意这里的准确说法是：

```text
不是复数 e^{iθ} 等于旋转矩阵，
而是“乘以 e^{iθ} 这个操作”在二维实向量空间中的矩阵表示是旋转矩阵。
```

### 3.2 RoPE 的核心公式

RoPE 不先把位置向量加到 token embedding 上，而是先计算：

```text
q_m = W_q x_m
k_n = W_k x_n
```

然后按位置旋转 q / k：

```text
q'_m = R(m * theta_i) q_m
k'_n = R(n * theta_i) k_n
```

其中 `m` 和 `n` 是 token 位置，`theta_i` 是第 `i` 个频率块的频率。

attention score：

```text
score(m, n) = (q'_m)^T k'_n
```

代入：

```text
score(m, n)
= (R(m * theta_i) q_m)^T (R(n * theta_i) k_n)
```

利用旋转矩阵性质：

```text
R(a)^T = R(-a)
R(a)R(b) = R(a + b)
```

得到：

```text
score(m, n)
= q_m^T R(m * theta_i)^T R(n * theta_i) k_n
= q_m^T R(-m * theta_i) R(n * theta_i) k_n
= q_m^T R((n - m) * theta_i) k_n
```

所以 RoPE 展开后，位置以相对距离的形式出现在 q/k 的比较规则中：

```text
R((n - m) * theta_i)
```

这不是说 APE 不能出现相对位置，而是 RoPE 把相对位置作为 q/k 点积中的整体几何调制方式。

## 4. APE 和 RoPE 的主要区别

### 4.1 一句话区别

```text
APE 加的是一个会旋转的位置向量；
RoPE 旋转的是 q/k 语义向量本身。
```

更完整地说：

```text
APE:
token_with_pos = token_embedding + PE(pos)
```

其中 `PE(pos)` 自身随位置变化而旋转，但 token embedding 本身没有被旋转。

RoPE:

```text
q_with_pos = R(pos) W_q token_embedding
k_with_pos = R(pos) W_k token_embedding
```

也就是说，RoPE 让 q/k 自身带上位置相位。

### 4.2 在 attention 中的区别

APE 的 score 展开后是：

```text
score(m, n)
= x_m^T A x_n
+ x_m^T A p_n
+ p_m^T A x_n
+ p_m^T A p_n
```

它的位置信息是作为额外特征混进 token 表示中的。

RoPE 的 score 是：

```text
score(m, n)
= q_m^T R((n - m) * theta_i) k_n
```

它的位置信息是作为 q/k 之间的相对几何变换进入点积中的。

因此两者的核心区别不是：

```text
APE 没有相对位置，RoPE 才有相对位置。
```

更准确的说法是：

```text
APE 的位置-位置项可以表达相对位置；
RoPE 则把相对位置结构直接放进 q/k 点积的整体比较方式里。
```

### 4.3 关于外推性的直觉

本文先不展开 RoPE 的长度外推问题。

对本项目来说，先理解 RoPE 如何把位置旋转作用到 q/k 上就够了。长度外推涉及训练长度、频率设置、上下文扩展等额外问题，可以单独放到后续文档里讲。

## 5. 本项目中的 RoPE 实现

两套实现均在各自的 `rope.py` 中提供相同的函数接口：

```python
def rope_exp(q, cur_valid_len=None, theta=10000):
```

输入 `q` 的形状约定为：

```text
(batch, head, time, head_dim)
```

其中：

```text
batch    : batch size
head     : attention head 数量
time     : 当前序列长度
head_dim : 每个 head 的维度
```

### 5.1 位置 position

如果是 prefill / 训练阶段，`cur_valid_len is None`，此时 `time = t`。Keras 实现为：

```python
position = K.arange(t, dtype="float32")[None, None, :, None]
```

形状是：

```text
(1, 1, time, 1)
```

它会广播到：

```text
(batch, head, time, head_dim/2)
```

PyTorch 使用 `torch.arange(..., device=q.device)` 构造相同的位置张量。

如果是 decode 阶段，当前输入通常只有一个 token，此时传入 `cur_valid_len`。Keras 实现为：

```python
t = cur_valid_len - 1
position = K.cast(K.reshape(t, (-1, 1, 1, 1)), dtype="float32")
```

形状是：

```text
(batch, 1, 1, 1)
```

PyTorch 对应使用 `reshape(...).to(dtype=torch.float32, device=q.device)`，形状和位置语义相同。

这里使用 `cur_valid_len - 1` 的前提是：

```text
cur_valid_len 表示写入当前 token 后的有效长度。
```

因此当前 token 的位置索引是：

```text
cur_valid_len - 1
```

这个约定需要和 KVCache 写入位置保持一致。

### 5.2 频率 theta

以下是 Keras 写法：

```python
index = K.arange(c // 2, dtype="float32")
m_theta = position * theta ** (-2 * index / c)
```

其中 `c = head_dim`。

对应数学公式：

```text
theta_i = theta^(-2i / c)
angle_i = position * theta_i
```

默认：

```text
theta = 10000
```

PyTorch 使用 `torch.arange(c // 2, ...)` 完成同一计算。

这和经典 sinusoidal position encoding 使用的频率形式一致。

### 5.3 维度配对方式

常见 RoPE 有两种配对方式。

第一种是相邻配对：

```text
(x_0, x_1), (x_2, x_3), ...
```

第二种是前后半区配对：

```text
(x_0, x_{c/2}), (x_1, x_{c/2+1}), ...
```

本项目的两个版本都使用第二种方式。Keras 写法是：

```python
left, right = tf.split(q, 2, axis=-1)
complex_q = tf.complex(left, right)
```

也就是：

```text
left  -> 复数实部
right -> 复数虚部
```

对应：

```text
z = left + i * right
```

PyTorch 对应使用 `torch.split()` 和 `torch.complex()`。

### 5.4 复数乘法实现旋转

Keras 实现为：

```python
rotate_q = complex_q * tf.exp(tf.complex(tf.zeros_like(m_theta), m_theta))
```

其中：

```text
tf.exp(i * m_theta) = cos(m_theta) + i sin(m_theta)
```

因此：

```text
complex_q * exp(i * m_theta)
```

PyTorch 对应使用 `torch.exp(torch.complex(...))`，复数乘法的数学含义不变。

等价于对每一组二维向量做旋转：

```text
real' = real * cos(angle) - imag * sin(angle)
imag' = real * sin(angle) + imag * cos(angle)
```

Keras 最后取出实部和虚部：

```python
real = tf.math.real(rotate_q)
imag = tf.math.imag(rotate_q)
```

并拼回原来的维度顺序：

```python
return K.concatenate([real, imag], axis=-1)
```

PyTorch 对应使用 `torch.real()`、`torch.imag()` 和 `torch.cat()`。

因此本项目输出的维度排列仍然是：

```text
[rotated_left, rotated_right]
```

不是相邻交错排列。

### 5.5 Keras 与 PyTorch 的代码对应关系

| 操作 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| 获取形状 | `K.shape(q)` | `q.shape` |
| 创建位置索引 | `K.arange` | `torch.arange` |
| 拆分最后一维 | `tf.split` | `torch.split` |
| 构造复数 | `tf.complex` | `torch.complex` |
| 复指数 | `tf.exp` | `torch.exp` |
| 取实部/虚部 | `tf.math.real/imag` | `torch.real/imag` |
| 拼接 | `K.concatenate` | `torch.cat` |

PyTorch 显式把临时张量放到 `q.device`；Keras 由张量运算和后端处理设备。除此之外，两套 `rope_exp()` 的输入输出、频率公式与 decode 位置约定一致。

### 5.6 使用注意点

`rope_exp()` 默认要求：

```text
head_dim 必须是偶数
```

因为最后一维需要被分成实部和虚部两半。

在 attention 中，RoPE 应该应用在 `q` 和 `k` 上：

```text
q = rope_exp(q, ...)
k = rope_exp(k, ...)
```

一般不应用在 `v` 上，因为 RoPE 的目标是改变 q/k 的匹配方式，而不是改变 value 的内容。

decode 阶段需要特别注意：

- 当前 token 的 position 要和 cache 写入位置一致。
- 如果 key 已经以 RoPE 后的形式写入 cache，后续取出时不要再次 RoPE。
- 如果 cache 存的是 RoPE 前的 key，则每次参与 attention 前需要保证 key 使用正确 position 旋转。

本项目的实现核心可以概括为：

```text
把 head_dim 的前半部分作为实部，后半部分作为虚部；
用 position * theta_i 构造旋转角；
通过复数乘 exp(i * angle) 完成 RoPE 旋转；
再把实部和虚部拼回原来的 tensor。
```
