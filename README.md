# 游戏视频 → Blender 白模实时建模工作流

依据《白模实时建模工作流设计文档.md》搭建的 P0 生产流水线：
游戏视频 → 切片/预检 → 语义解析（场景资产库）→ 相机解算 → Blender 白模 → 配对合成 → 质检交付。

## 环境要求

- Python 3.10+，`pip install -r requirements.txt`
- ffmpeg / ffprobe（PATH 中可用）
- Blender 4.x（PATH 或 config.yaml `blender.executable`）
- 可选（GPU ≥8GB）：`pip install torch transformers ultralytics`，SAM2 按官方仓库安装，
  checkpoint 放 `checkpoints/`；MASt3R/GLOMAP clone 到 `external/`
- Claude API：设置环境变量 `ANTHROPIC_API_KEY`

## 快速开始（单片段 → 成对数据）

```powershell
# Stage 1：切片（15-30s 一镜到底，切镜段自动回退）
python -m src.capture.slicer --input data/raw/game.mp4

# Stage 2-7 全流程（断点续跑，失败阶段可用 --stages 单独重跑）
python run_pipeline.py --clip game_seg000

# 指定阶段 / 强制重跑
python run_pipeline.py --clip game_seg000 --stages semantic,world,generate,compose,qa --force
```

交付目录：`data/output/<clip_id>/`
（`白模视频.mp4`、`目标视频.mp4`、`对齐映射说明.json`、`side_by_side.mp4`）

## 阶段与代码对照

| 阶段 | 模块 | 说明 |
|---|---|---|
| Stage 1 切片/门禁 | `src/capture/slicer.py`、`src/preprocess/purity.py` | PySceneDetect 切镜检测；UI/水印打回 |
| Stage 2 预处理 | `src/preprocess/preprocess.py` | 全帧 + 自适应关键帧（0.5fps 起步）、音频、meta.json |
| Stage 3 语义解析 | `src/vision/semantic_parse.py`、`asset_store.py`、`batch_parse.py` | Claude API + schema 校验 + SQLite 跨帧资产库；多人同框打回 |
| Stage 3 本地视觉 | `src/vision/depth.py`、`segment.py`、`pose.py` | Depth Anything V2 / SAM2 / YOLO-pose，未安装优雅跳过 |
| Stage 4 相机解算 | `src/sfm/camera_solve.py`、`scale_ground.py` | MASt3R/GLOMAP → 位姿内插；人物尺度锚点；地面拟合；世界落位与姿态提升 |
| Stage 5 白模生成 | `src/blender_gen/gen_blockout.py`（Blender 内执行）、`generate.py` | 图元白模 + Armature/步态动画 + 碰撞体 + 合规校验 + EEVEE 渲染 |
| Stage 6 配对合成 | `src/render/compose.py` | 白模序列 + 原 audio → 白模视频；交付目录 |
| Stage 7 质检 | `src/qa/qa_check.py` | 对齐/运动同步/纯净度/合规/音频自动检查 |
| 编排 | `src/pipeline.py`、`run_pipeline.py` | 断点续跑（`_done_<stage>.marker`） |
| 自研 MCP | `tools/vision_mcp/server.py` | `python -m tools.vision_mcp.server`（stdio） |

## MCP（设计文档 §5）

`.mcp.json` 已配置：`blender`（需在 Blender 内安装 ahujasid/blender-mcp 插件）、
`filesystem`、`memory`、`vision`（自研）。调试期交互驱动，生产期走 Python 流水线。

## Skills（设计文档 §6）

`.claude/skills/`：video-prep / scene-parse / camera-solve / blockout-gen / pair-qa，
每个含触发条件、输入输出、命令步骤、引用标准与失败处理。

## 当前实现状态（诚实说明）

- MASt3R/GLOMAP 未配置时相机自动降级为恒定机位（低置信度），白模可产出但视角不准 —— 正式交付需配置真实解算后端
- RTMPose/WHAM 后端预留接口，当前姿态用 YOLO11-pose；无姿态数据时人物用内置 walk_cycle 步态
- 纯净度检测以启发式为主，安装 paddleocr 后增强 OCR 检测
- 里程碑建议按设计文档 §10：M0 环境（本仓库即绪）→ M1 单片段跑通 → M2 小批量 → M3 全自动 + Batch API
