# LFMN 图像超分复现与研究

这个仓库保存 LFMN 的可运行复现代码、公开预训练权重、评测脚本、研究笔记和创新实验记录。原项目代码位于 [`LFMN/`](LFMN/)，模型结构见 [`LFMN/model/lfmn.py`](LFMN/model/lfmn.py)。本仓库的研究方向和现阶段实验假设见 [`创新思路与实验记录.md`](创新思路与实验记录.md)，阶段计划见 [`研究进度计划.md`](研究进度计划.md)。

`datasets/`、`experiment/`、缓存、参考仓库副本和本地论文文件不会上传。请自行准备合法获取的 DIV2K 训练集及标准 benchmark 数据，并按 [`LFMN/README.md`](LFMN/README.md) 中的说明布置目录。已有评测记录是预训练权重的复现结果；新增结构尚未获得实验增益，不应引用为新模型结果。

本目录作为整体 Git 仓库。`LFMN/` 原有 Git 元数据保存在本地 `LFMN/.git_upstream_backup/`，不会同步到 GitHub。每次由 Codex 修改并验证的代码或研究文档应独立提交，提交说明写明改动目的与验证情况；如果 GitHub 推送失败，本地提交仍保留，并在答复中说明。

LFMN 原始项目来自 [hehesjtu/LFMN](https://github.com/hehesjtu/LFMN)。原始方法和预训练权重应归属并引用原作者；本仓库不声称它们为原创。

第一轮数据下载和核验步骤见 [数据集下载与首轮实验准备.md](数据集下载与首轮实验准备.md)。
