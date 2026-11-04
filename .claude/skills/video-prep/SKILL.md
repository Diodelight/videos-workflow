---
name: video-prep
description: 新游戏视频进入白模流水线时的切片（15–30s 一镜到底）、切镜检测、纯净度预检、抽帧与音频提取。当用户要求处理/切分/预检新的游戏视频、或准备流水线输入数据时使用。
---

# video-prep —— 采集切片与预处理（Stage 1–2）

## 触发条件
- 用户提供新的游戏实录视频，要求进入白模流水线
- 要求切片、抽帧、音频提取、纯净度预检（字幕/水印/UI 检测）

## 输入输出约定
- 输入：`data/raw/<name>.mp4`（原生 ≥720p、≥24fps、一镜到底优先）
- 输出：`data/clips/<name>_segNNN.mp4`（15–30s）、`data/clips/manifest.json`、
  `data/analysis/<clip_id>/`（全帧、关键帧、audio.m4a、meta.json、purity.json）

## 步骤
1. 确认源视频已放入 `data/raw/`，分辨率/帧率/时长满足 §2.1 要求
2. 切片（PySceneDetect 检出切镜的段自动回退；>30s 一镜到底按 30s 截断）：
   ```powershell
   python -m src.capture.slicer --input data/raw/<name>.mp4
   ```
3. 逐片段纯净度预检（FAIL 即打回，不进入后续环节）：
   ```powershell
   python -m src.preprocess.purity --clip <clip_id>
   ```
4. 预处理（全帧 + 自适应关键帧 + 音频 + 元数据）：
   ```powershell
   python -m src.preprocess.preprocess --clip <clip_id>
   ```
5. 也可一条命令跑完 2–4 步：`python run_pipeline.py --clip <clip_id> --stages preprocess,purity`
6. 检查 `meta.json`：frame_count / fps / width / height 是后续对齐基准，须口头确认无 UI 打回项

## 引用标准
《白模实时建模工作流设计文档》§2.1（时长/纯净度/禁止项）、§4.1、§4.2

## 失败处理
- `ModuleNotFoundError: scenedetect` → `pip install scenedetect`
- 纯净度 FAIL → 向用户说明打回原因（reasons 字段），更换素材或人工裁剪 UI 区域
- OCR 未安装 → purity 以启发式运行（method=heuristic_only），建议安装 paddleocr 加强
- 无音轨 → meta.has_audio=false，后续合成无音频（记录，不阻塞）
