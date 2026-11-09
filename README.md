# 情感分类器作业 (Sentiment Classifier)

## 研究问题 (Research Question)
本项目旨在构建一个二元逻辑回归分类器，用于区分中文文本的正面（Positive）与负面（Negative）情感，并探讨该模型在处理复杂修辞（如反讽）及长篇小说宏观情感走向时的表现与局限性。

## 标签定义 (Label Definition)
- **Label 0 (负面)**：豆瓣影评中的 1 星评价。
- **Label 1 (正面)**：豆瓣影评中的 5 星评价。
*(注：已剔除 2-4 星的中性评价以保证二分类的纯粹性)*

## 数据来源 (Data Sources)
1. **训练/测试数据**：课程提供的 `douban.json`（存放于 `data/` 目录）。
2. **应用测试文本**：老舍长篇小说《骆驼祥子》全文（存放于 `data/luotuoxiangzi.txt`）。

## 如何执行代码 (How to run)
在终端中运行以下命令：
```bash
python train.py
