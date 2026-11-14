# 4Cam — 膝关节 6DOF 四相机光学测量仿真

本仓库是四相机路线的唯一仿真代码库：**2 台 Basler acA2040-90umNIR + 2 台海康 MV-CU020-90UM**，
用于回答三个工程问题：

1. 四台异构相机分别摆在哪、多高、朝哪个方向看；
2. 实验室步道和分析窗口要多大，受试者从哪里起步、走到哪里；
3. 在给定布局下，像素噪声、同步误差、标定残差会把 6DOF 精度推到什么程度。

完整方案与推导见 [`sim/docs/design.md`](sim/docs/design.md)。

## 硬件与仿真口径

| 项目 | 设定 |
|---|---|
| 相机 | 2 × Basler acA2040-90umNIR（2048×2048，5.5 µm）<br>2 × 海康 MV-CU020-90UM（2048×1200，4.0 µm） |
| 镜头 | 12 mm（Basler 50.3°×50.3°，海康横放 37.7°×22.6°、竖放 22.6°×37.7°） |
| 帧率 / 光源 | 90 Hz，850 nm 近红外 |
| 标记点 | 每侧 12 点，双膝共 24 点 |
| 步态数据 | OpenSim Gait2392，1 个周期（51 帧），前进 1.3 m |
| 场地 | 步道 4.0 m × 1.2 m，相机高度不设硬限制 |
| 精度目标 | 0 失败帧；旋转 ≤0.5°，平移 ≤0.5 mm |
| 误差源 | 像素噪声、标定残差、同步误差（不含软组织伪影） |

## 目录结构

```text
4Cam/
├── README.md
└── sim/
    ├── sim/
    │   ├── trajectory.py     # Gait2392 轨迹加载与标记点
    │   ├── camera.py         # CameraSpec：型号/传感器/焦距/横竖放/FOV
    │   ├── detection.py      # 视锥、最小成像尺寸、掠射角、胶囊遮挡
    │   ├── reconstruct.py    # 多视图 DLT + 加权高斯-牛顿、三维精度场
    │   ├── kinematics.py     # 刚体 SVD 拟合 + Grood & Suntay 分解
    │   ├── pipeline.py       # 覆盖统计、失败帧、端到端 6DOF 蒙特卡洛
    │   ├── layout.py         # 4 相机模板搜索、解析误差、局部优化
    │   ├── search.py         # 通用候选缓存与评分
    │   ├── walkway.py        # 步道几何、分析窗口、相机位姿换算
    │   └── report.py         # 布置表、俯视图/侧视图、覆盖热图
    ├── scripts/
    │   ├── search_layout.py  # 主入口：布局搜索 + 蒙特卡洛 + 出报告
    │   └── fov_requirement.py# 各机位所需最小 FOV 预筛
    ├── tests/                # 27 个单元测试
    ├── data/                 # trajectory_cache.mat（OpenSim 轨迹缓存）
    ├── results/              # 最近一轮布局结果与图
    └── docs/design.md        # 方案、误差模型、复用评估、结果与局限
```

## 快速开始

```bash
pip install -r sim/requirements.txt

cd sim
python -m pytest tests -q
python scripts/search_layout.py --travel 1300 --sides RL --sigma 0.3 --top-k 3 --mc-n 200
```

`search_layout.py` 的主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--travel` | 1300 | 一个周期前进距离（mm） |
| `--sides` | RL | `R` / `L` / 双膝 |
| `--sigma` | 0.3 | 全局像素噪声 σ（px） |
| `--distances` | 2500…4500 | 相机距离候选（mm） |
| `--heights` | 300…2400 | 相机高度候选（mm） |
| `--top-k` / `--mc-n` | 5 / 80 | 局部优化数量、蒙特卡洛次数 |

输出到 `sim/results/`：

| 文件 | 内容 |
|---|---|
| `setup_sheet.csv` | 实验室摆位表：型号、镜头、朝向、坐标、方位角、距离、高度、瞄准点、FOV |
| `layout_recommended.json` | 完整布局与步道几何（程序可读） |
| `summary.txt` | 失败帧、解析误差、逐标记覆盖、蒙特卡洛结果 |
| `top_view.png` | 俯视图：步道、分析窗口、相机、光轴、覆盖足迹 |
| `side_view.png` | 侧视图：相机高度与俯仰 |
| `coverage_map.png` | 每标记 ≥2 视图的覆盖热图 |

## 当前结论（第一轮）

- **布局**：四角斜向。C1 Basler −45°/3.0 m/h 0.3 m，C2 Basler +45°/3.0 m/h 0.6 m，
  C3 海康 −135°/3.5 m/h 0.3 m，C4 海康 +135°/4.0 m/h 0.3 m，全部瞄准 (0, 550, 0) mm。
- **步道**：4.0 m × 1.2 m，分析窗口 2.2 m（X −1100 ~ +1100 mm），进入/退出各 0.9 m。
- **精度**：分析窗口内 0 失败帧；σ = 0.3 px 时旋转最差 0.453°、AP 平移 0.508–0.528 mm；
  σ = 0.1 px 时 AP 平移约 0.17 mm。
- **瓶颈**：内侧膝/胫后标记覆盖率仅 0.667；同步误差与标定残差尚未计入主结果。

## 后续路线

1. 同步误差（0.2 / 0.4 / 1.0 ms）与标定残差（0.3 / 0.5 mm）敏感性扫描；
2. 实测海康的 σ_px、850 nm 灵敏度、USB3 90 Hz 全画幅能力；
3. 提高内侧标记覆盖率，把 σ = 0.3 下的 AP 平移压到 0.5 mm 以内；
4. 可选：5 相机对照、1.5 / 1.9 m 身高缩放、3 步连续采集。
