# code_templates — 动态技能记忆库

本目录用于存放 Agent **成功执行过**的空间分析 Python 脚本（先尝试抽象化，失败则保存原始代码）。

- **registry.json**：模板索引，记录每个脚本的 `filepath`、`spatial_predicate`、`analytical_method`、`description` 等，供 LLM2Code 节点检索 Few-shot 使用。
- 成功执行后的脚本会以 `{method}_{timestamp}.py` 形式保存于此，并自动更新 `registry.json`。
- 每次 LLM2Code 成功时，**workspace** 目录下会同时生成 `last_successful_script.py`，即本次运行的完整脚本副本，便于直接查看或调试。

请勿手动删除正在被引用的模板文件，以免影响后续检索。
