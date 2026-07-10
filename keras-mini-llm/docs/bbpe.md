# BBPE 原理与实现

本文记录本项目中 Byte-level BPE tokenizer 的实现思路，对应代码主要在：

- `bbpe_trainer.py`
- `tokenizer.py`

项目里的 tokenizer 分成两部分：

- `BBPETrainer`：负责从纯文本中训练 vocab 和 merge rules。
- `Tokenizer`：负责在模型训练和推理阶段加载 vocab / merge rules，并完成 encode / decode。

## 1. 为什么使用 Byte-level BPE

BPE 的核心目标是把文本压缩成更短的 token 序列。它从最基础的符号开始，不断寻找语料中出现频率最高的相邻 token pair，并把这个 pair 合并成一个新 token。

Byte-level BPE 的起点不是字符，而是 UTF-8 byte：

```text
initial vocab: 0, 1, 2, ..., 255
```

这样做有两个好处：

- 不需要提前定义字符表，任意 UTF-8 文本都能编码。
- 遇到生僻字、符号、混合语言时，也可以退回 byte 表示，不容易出现无法编码的问题。

训练完成后，vocab 中的每个 token id 都对应一段 byte 序列：

```python
{
    0: [0],
    1: [1],
    ...
    256: [token_a_bytes + token_b_bytes],
    ...
}
```

decode 时只需要把 token id 映射回 byte 序列，再拼成 bytes 并按 UTF-8 解码。

## 2. 整体训练流程

`bbpe_trainer.py` 的训练主流程是：

```text
load_segments()
  -> build_linked_segments()
  -> train()
  -> save()
```

### 2.1 load_segments

`load_segments()` 负责读取文本并切分 segment。

代码中使用的 regex pattern 是：

```python
r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+[\s]*"""
```

这个 pattern 会把文本切成几类片段：

- 英文缩写后缀，例如 `'s`、`'t`、`'re`
- 字母片段
- 数字片段
- 标点或其他非空白符号片段

每个 segment 会统计频数：

```python
segment2freq[segment] += 1
```

后续训练时，相同 segment 只保留一份链表结构，并通过 `segment.freq` 表示它在语料里出现了多少次。

### 2.2 build_linked_segments

`build_linked_segments()` 会把每个 segment 编码成 UTF-8 bytes，然后转成双向链表。

例如：

```text
"你好" -> UTF-8 bytes -> [228, 189, 160, 229, 165, 189]
```

链表结构大致是：

```text
228 <-> 189 <-> 160 <-> 229 <-> 165 <-> 189
```

每个节点是一个 `Node`：

```python
class Node:
    __slots__ = ["value", "pre", "next", "segment"]
```

每个 segment 对应一个 `Segment`：

```python
class Segment:
    __slots__ = ["head", "tail", "freq"]
```

链表的好处是：当某个 pair 被合并时，只需要修改局部节点的前后指针，不需要频繁创建完整的新 list。

### 2.3 初始化 pair 统计

构建链表时，同时初始化三个核心结构：

```python
pair2freq
pair2nodes
freq2pair_heap
```

它们的含义是：

```text
pair2freq[pair]      -> 这个 pair 在整个语料中的加权频数
pair2nodes[pair]     -> 这个 pair 出现在哪些链表节点位置
freq2pair_heap       -> 用 heap 快速取出当前最高频 pair
```

这里的频数不是简单加 1，而是加上 segment 的出现频数：

```python
pair2freq[pair] = pair2freq.get(pair, 0) + segment.freq
```

如果一个 segment 出现了 100 次，那么它内部的一个 pair 对总频数的贡献就是 100。

## 3. 为什么不用每轮全量扫描

最朴素的 BPE 训练方式是：

```text
每一轮：
  1. 扫描所有 token 序列
  2. 统计所有 pair 频数
  3. 找出最高频 pair
  4. 全量替换这个 pair
```

这种方式容易理解，但每轮都全量扫描语料，语料稍大就会慢。

本项目的实现思路是：

```text
只在初始化时全量统计一次；
之后每次 merge，只更新受影响的局部 pair。
```

一次 pair merge 只会影响三个位置：

```text
left pair   : (left, old_a)
current pair: (old_a, old_b)
right pair  : (old_b, right)
```

合并成新 token 后，对应变成：

```text
new left pair : (left, new_token)
new right pair: (new_token, right)
```

所以不需要重新扫描整个语料，只需要更新这些局部 pair 的频数和位置索引。

## 4. 核心数据结构

### 4.1 Node

`Node` 是链表节点：

```python
class Node:
    __slots__ = ["value", "pre", "next", "segment"]
```

字段含义：

```text
value   -> 当前 token id
pre     -> 左侧节点
next    -> 右侧节点
segment -> 当前节点属于哪个 Segment
```

`segment` 反向引用很关键。通过 `pair2nodes[pair]` 找到某个 pair 的节点后，可以直接知道它属于哪个 segment，并根据 `segment.freq` 更新加权频数。

### 4.2 Segment

`Segment` 表示一个去重后的文本片段：

```python
class Segment:
    __slots__ = ["head", "tail", "freq"]
```

字段含义：

```text
head -> 当前链表头节点
tail -> 当前链表尾节点
freq -> 这个 segment 在语料中出现的次数
```

当 merge 发生在头部或尾部时，需要更新 `segment.head` 或 `segment.tail`。

### 4.3 pair2freq

`pair2freq` 记录每个 pair 的全局加权频数：

```python
pair2freq[(token_a, token_b)] = freq
```

训练时每一轮都选择当前频数最高的 pair 进行合并。

### 4.4 pair2nodes

`pair2nodes` 记录 pair 出现在哪些节点位置：

```python
pair2nodes[(token_a, token_b)] = {node_1, node_2, ...}
```

这里保存的是 pair 的左节点。对于链表：

```text
A <-> B <-> C
```

pair `(A, B)` 的位置保存的是节点 `A`。

### 4.5 freq2pair_heap

`freq2pair_heap` 用来快速取最高频 pair：

```python
freq2pair_heap = [(-freq, pair), ...]
```

Python 的 `heapq` 是小根堆，所以代码里存的是负频数。

## 5. Lazy heap update

`heapq` 不支持直接修改堆中某个 pair 的频数。常规做法是：每次频数变化时，直接 push 一个新的 `(-freq, pair)`。

这样堆里会存在旧记录。取最大 pair 时再判断这条记录是否仍然有效：

```python
freq, pair = heapq.heappop(self.freq2pair_heap)
freq = -freq

if freq == self.pair2freq.get(pair, 0):
    return freq, pair
```

如果 heap 中的频数和 `pair2freq` 当前值不一致，说明它是旧记录，直接丢掉。

这就是 lazy heap update：

```text
更新时不删除旧记录；
取出时再判断是否过期。
```

它避免了频繁重建 heap，也避免了在 heap 中定位某个旧 pair 的复杂操作。

## 6. update() 做了什么

`update(pair, current_vocab_index)` 是训练中最核心的函数。

假设当前要合并的 pair 是：

```text
(A, B) -> X
```

在链表中可能出现为：

```text
L <-> A <-> B <-> R
```

合并后变成：

```text
L <-> X <-> R
```

这一步至少要同步更新六类状态：

```text
1. vocab
2. merge_rules
3. 链表节点指针
4. pair2freq
5. pair2nodes
6. freq2pair_heap
```

### 6.1 新增 vocab 和 merge rule

代码会先记录合并规则：

```python
rule = (pair, new_value)
self.merge_rules.append(rule)
self.vocab[new_value] = self.vocab[value_1] + self.vocab[value_2]
```

这里 `vocab[new_value]` 保存的是合并后 token 对应的 byte 序列。

### 6.2 找到受影响的 Segment

`pair2nodes[pair]` 里保存的是这个 pair 的所有出现位置。代码先根据这些 node 找到对应的 segment：

```python
head2segments = set()
for node in self.pair2nodes[pair]:
    head2segments.add(node.segment)
```

然后只遍历这些受影响的 segment，而不是遍历所有 segment。

### 6.3 更新当前 pair

当找到 `(A, B)` 后，当前 pair 的频数要减少：

```python
self.pair2freq[pair] -= segment.freq
```

如果减少后频数仍然大于 0，就把新频数 push 回 heap，并从 `pair2nodes[pair]` 中删掉当前位置。

如果频数已经为 0，就从 `pair2freq` 和 `pair2nodes` 中删除这个 pair。

### 6.4 更新左侧 pair

如果当前 pair 左边还有节点 `L`：

```text
L <-> A <-> B
```

合并后：

```text
L <-> X
```

所以旧 pair `(L, A)` 要减少，新 pair `(L, X)` 要增加。

对应代码逻辑是：

```text
old_left_pair = (L, A)  -> freq -= segment.freq
new_left_pair = (L, X)  -> freq += segment.freq
```

同时还要维护 `pair2nodes` 中的出现位置。

### 6.5 更新右侧 pair

如果当前 pair 右边还有节点 `R`：

```text
A <-> B <-> R
```

合并后：

```text
X <-> R
```

所以旧 pair `(B, R)` 要减少，新 pair `(X, R)` 要增加。

对应逻辑是：

```text
old_right_pair = (B, R) -> freq -= segment.freq
new_right_pair = (X, R) -> freq += segment.freq
```

### 6.6 更新链表指针

新建节点 `X` 后，需要接回原链表。

如果右边有 `R`：

```python
node.next = curr.next.next
node.next.pre = node
```

否则说明 merge 发生在尾部：

```python
segment.tail = node
```

如果左边有 `L`：

```python
node.pre = curr.pre
node.pre.next = node
```

否则说明 merge 发生在头部：

```python
segment.head = node
```

最后：

```python
curr = node.next
```

这里不是随便写的。合并完成后，继续从新节点右侧往后扫描，可以避免重复使用已经被合并掉的节点，也能正确处理相邻重复 pair 的边界。

## 7. 重复 pair 的边界问题

重复字符或重复 byte 场景是 BPE 实现里很容易写错的地方。

例如：

```text
A A A A A
```

如果当前要合并 `(A, A)`，不能简单地同时合并所有看起来相邻的位置，因为这些 pair 会重叠：

```text
(A1,A2), (A2,A3), (A3,A4), (A4,A5)
```

一次从左到右的合法合并应该类似：

```text
A A A A A
-> X X A
```

而不是错误地让 `A2` 同时参与两次合并。

因此 `update()` 中需要在链表上按顺序扫描，并在每次合并后移动到 `node.next`，这样已经被合并掉的节点不会再次参与当前轮 merge。

这也是为什么这段逻辑不适合拆成太多小函数：链表位置、pair 频数、pair 位置索引和 heap 状态是同时变化的，拆得过碎反而容易漏掉边界同步。

## 8. train() 的终止条件

训练从 byte vocab 之后开始：

```python
current_vocab_index = 256
```

每一轮：

```text
1. 从 heap 中取出最高频 pair
2. 如果没有 pair，停止
3. 如果最高频低于 min_pair_freq，停止
4. 合并 pair，新增 token
5. current_vocab_index += 1
```

训练最多到：

```python
max_vocab_size
```

代码里还有一个小的维护动作：

```python
if current_vocab_index % 1000 == 0:
    self.create_freq2pair_heap()
```

因为 lazy heap 会积累过期记录，定期重建 heap 可以清理旧记录。

## 9. 保存格式

训练结束后会保存两个文件：

```text
tokenizer_config/vocab.json
tokenizer_config/merge_rules.json
```

`vocab.json` 保存 token id 到 byte 序列的映射：

```json
{
    "0": [0],
    "1": [1],
    "256": [228, 189]
}
```

`merge_rules.json` 保存按训练顺序得到的 merge 规则：

```json
[
    [[228, 189], 256],
    [[256, 160], 257]
]
```

这里的规则顺序很重要。encode 时如果多个 pair 都能合并，应该优先应用训练时更早出现的规则。

## 10. tokenizer.py 中的 encode

`tokenizer.py` 是正式用于训练模型和推理的 tokenizer。

初始化时会加载 vocab：

```python
self.vocab = {int(i): j for i, j in json.load(f).items()}
```

然后追加特殊 token：

```python
for token in ["<bos>", "<eos>", "<unk>", "<pad>"]:
    self.vocab[len(self.vocab)] = list(token.encode("utf8"))
    self.special_ids[token] = len(self.vocab) - 1
```

这里有一个明确约定：特殊 token 只通过代码显式拼接 id 进入序列，不通过文本字符串进入 tokenizer。

也就是说，训练和推理代码应该写成：

```python
ids = tokenizer.encode_text(text)
ids = ids + [tokenizer.special_ids["<eos>"]]
```

而不是：

```python
ids = tokenizer.encode_text(text + "<eos>")
```

如果原始文本中真的出现 `"<eos>"`、`"<pad>"` 这类可见字符串，它们会被当作普通文本做 BBPE 编码，不会被识别成控制 token。

因此本项目中的特殊 token 规则是：

```text
文本字符串里的 <eos> : 普通文本
special_ids["<eos>"] : 控制 token
```

这样可以让 BBPE 训练保持纯文本逻辑：特殊 token 不参与 vocab/merge rules 训练，也不会进入普通 byte pair merge。

merge rules 会被转换成带 rank 的 dict：

```python
merge_rules[tuple(pair)] = [tuple(pair), new_token_id, i]
```

其中 `i` 就是这条规则的 rank，越小表示越早训练出来，优先级越高。

### 10.1 encode_text

完整文本入口是：

```python
encode_text(text)
```

它会先用同样的 regex pattern 切分文本：

```python
segments = re.findall(self.pattern, text)
```

然后对每个 segment 单独调用：

```python
self.encode(segment)
```

### 10.2 encode

`encode(segment)` 的流程是：

```text
1. segment -> UTF-8 bytes
2. 找出当前所有相邻 pair
3. 在这些 pair 中找到 rank 最小的 merge rule
4. 从左到右应用这条 merge rule
5. 重复，直到没有可用 merge rule
```

这里和 `BBPETrainer.encode()` 的表现形式不完全一样：

- `BBPETrainer.encode()` 是按 `self.merge_rules` 顺序逐条扫描，主要用于训练后自测。
- `Tokenizer.encode()` 是每轮从当前 pair 中找 rank 最小的规则，更适合正式 encode。

两者遵循的核心原则是一致的：越早训练出来的 merge rule，优先级越高。

## 11. decode

decode 很直接：

```python
new_token_ids = []
for iid in token_ids:
    new_token_ids.extend(self.vocab[iid])

return bytes(new_token_ids).decode("utf8", errors="ignore")
```

因为每个 token id 最终都能还原成 byte 序列，所以 decode 不需要知道 merge 过程，只需要查 vocab。

## 12. 当前实现的取舍

这个 BBPE 实现的重点是把训练逻辑写清楚，同时避免最朴素全量扫描的低效。

它做了这些优化：

- 使用 byte-level vocab，保证任意 UTF-8 文本可编码。
- 使用 regex 先切 segment，减少跨类别合并。
- 对重复 segment 计频，避免重复保存相同链表。
- 使用双向链表维护 token 序列，merge 时局部改指针。
- 使用 `pair2freq` 维护全局 pair 频数。
- 使用 `pair2nodes` 定位 pair 出现位置。
- 使用 lazy heap 快速取最高频 pair。

后续还可以继续完善：

- 为 `update()` 增加更细的单元测试。
- 单独测试重复 pair 场景，例如 `aaaaa`。
- 增加小语料的可复现实验，方便读者快速跑通。

## 13. 一句话总结

这个 BBPE 实现不是简单地每轮重新扫描全量文本，而是把语料表示成链表，并维护 pair 频数、pair 位置索引和 lazy heap。

每次 merge 只更新局部受影响的 pair，因此更接近一个真正可扩展的 BPE trainer，而不是纯教学版的暴力实现。
