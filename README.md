# PolicyInduction（Boosting 版）

可解释的二分类器：LLM 写自然语言规则，[TypeSafe Jev](https://docs.typesafe.ai) 给每个样本打"规则成立的概率"，L1 逻辑回归学习每条规则的权重。规则通过 **boosting** 逐轮发现：每一轮都针对当前模型判错的样本写新规则，并且只保留在 LLM 从未见过的数据上确实降低误差的规则。

## 原理

```
训练集 ──┬── P 展示池（30%）：LLM 只能看到这里的样本和标签
         └── V 验证池（70%）：LLM 永远看不到；所有决策只依据这里

第 0 轮：从 P 抽 YES/NO 各 20 条 → LLM 写种子规则
第 1…R 轮：
  1. 在当前规则上做交叉验证 → 每个样本的 out-of-fold P(YES)
  2. 残差 g = y − p（只用 P 的样本挑错例）
  3. 奇数轮给 LLM 看漏判的 YES，偶数轮看误判的 NO，每次都附带判对的同类样本作对照
  4. LLM 提出新规则 → Jev 在全部样本上打分
  5. 过滤：常数规则、与已有规则高度相关、只在 P 上有效（在 V 上无效）
  6. 逐条尝试：只有当 V 上逐样本 log-loss 的平均改进 > 1 个标准误时才接受
  7. 连续 2 轮没有新规则被接受，或规则数达到上限时停止
收尾：按一倍标准误规则选 C → 在拼起来的 V out-of-fold 概率上选阈值
预测：15 个折模型的平均概率 ≥ 阈值 → YES
```

- **规则只加不删**：被后来的规则取代的旧规则，由 L1 把权重压到 0。
- **特征是概率**：Jev 的 `noul`（P(规则成立)），不做二值化。
- **规则数上限** = `min(max_policy_length, 少数类样本数 / 10)`。
- **Jev 版本锁定**：第一次请求时确定具体版本（如 `jev-1.13.0`），写进模型文件，预测时强制使用同一版本。

## 安装

```bash
cd policy-induction
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

以可编辑模式安装后，`experiments/` 下的脚本可以直接 `import policy_induction`。只需要运行时依赖的话，也可以用 `requirements.txt`。

## 配置 `.env`

| 变量 | 用途 |
|---|---|
| `TYPESAFE_API_KEY` | **必需**，Jev 打分。在 <https://console.typesafe.ai/keys> 创建 |
| `GOOGLE_AI_API_KEY` | 使用 `gemini-*` 生成规则时需要 |
| `OPENAI_API_KEY` | 使用 `gpt-*` 生成规则时需要 |

复制 `.env.example` 为 `.env` 并填写。`.env` 已加入 `.gitignore`，不会被提交。

## 使用

```python
import asyncio
from dotenv import load_dotenv
from policy_induction import PolicyInduction, WeightConfig

load_dotenv()

async def main():
    model = PolicyInduction(
        task_description="Predict whether reply A changed the original poster's view. YES = A won.",
        gen_model="gemini-3.5-flash",
        max_policy_length=30,
        weight_config=WeightConfig(beta=0.5),
        save_path="runs/cmv",
    )
    await model.fit(X_train, y_train)          # y: "YES"/"NO" 或 1/0
    print(model.rule_table())                  # 规则、权重、成立比例
    labels = await model.predict(X_test)       # ["YES", "NO", ...]
    probs = await model.predict_proba(X_test)
    model.save()                               # model.json, models.joblib, report.md
    await model.aclose()

asyncio.run(main())
```

加载已保存的模型：`PolicyInduction.load("runs/cmv")`。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `task_description` | 必填 | 预测什么、YES/NO 的含义。对规则质量影响最大 |
| `gen_model` | `gemini-3.5-flash` | 生成规则的 LLM（`gemini-*` / `gpt-*`，或自定义 `RuleGenerator`） |
| `max_policy_length` | 30 | 规则数上限（≤100），还会受数据量限制 |
| `gen_temperature` | 1.0 | 生成时的随机性 |
| `random_state` | 0 | P/V 划分、样本抽取、交叉验证折。**不控制 LLM** |
| `weight_config` | `WeightConfig()` | `beta`（只用于选阈值）、`Cs`、`cv_folds`、`cv_repeats`、`one_se_rule`、`class_weight_balanced` |

`BoostConfig` 存放 boosting 的内部常数（P 占比、每轮规则数、过滤阈值、早停等），一般不需要改。

## 输出

`save()` 会在 `save_path` 下写出：

- `model.json`：规则、阈值、锁定的 Jev 版本、指标、每轮日志
- `models.joblib`：15 个折模型（预测时取平均）
- `report.md`：可读报告，包括规则按权重排序、验证指标、每轮 boosting 的记录
- `jev_cache.sqlite`：Jev 答案缓存（按 版本 + 样本 + 规则 缓存），中断后重跑只补缺失部分
- `checkpoint.json`：训练中途的状态，训练完成后自动删除

报告里的指标都是 **V 上的 out-of-fold 结果**，也就是在生成规则的 LLM 没见过的样本上的表现。

## 规则的写法

Jev 按字面意思逐条判断规则，并且不擅长数值计算、多个条件和多步推理。生成 prompt 因此要求每条规则：

- 只描述一个可观察的条件（不能用"且/或"组合）
- 只描述条件，不写结论（方向由权重决定）
- 用反引号引用字段名，例如 `` `argument_A` ``
- 不引用样本原句，不出现具体人名、公司名、数字

## 已知限制

- **数值型数据**：Jev 不擅长数值比较。以数值列为主的表格数据（如 COMPAS），效果可能不如直接用逻辑回归。
- **成对比较任务**（如 CMV 的 A vs B）：目前按普通样本处理。反对称特征（f(A) − f(B)）尚未实现。
- **数据量**：V 需要足够大，接受检验才有统计效力。只有几百行时，规则上限会被自动压低。
- **生成仍有随机性**：比较不同配置时至少跑 3 个 `random_state`。

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试完全离线：用一个假的生成器和假的 Jev，在合成数据上验证 boosting 能找出全部信号规则、V 的样本从不出现在 prompt 里、断点续跑、保存和加载、规则上限等。

## 目录

```
policy_induction/
  model.py       PolicyInduction：boosting 循环、最终拟合、预测、保存和加载
  weights.py     交叉验证、选 C、配对接受检验、阈值
  scorer.py      Jev 打分、版本锁定、SQLite 缓存
  generator.py   生成 LLM（Gemini / OpenAI）
  prompts.py     生成 prompt 和规则写法约束
  config.py      WeightConfig、BoostConfig
experiments/     各数据集的实验脚本（data/ 和结果不入库）
tests/
pyproject.toml
```
