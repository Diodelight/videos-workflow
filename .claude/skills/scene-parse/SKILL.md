---
name: scene-parse
description: 游戏关键帧语义解析（Claude API）与跨帧一致的场景资产库构建。当用户要求解析场景、构建/修正场景 JSON、处理对象 ID 漂移或多人同框打回时使用。
---

# scene-parse —— 关键帧语义解析（Stage 3）

## 触发条件
- 需要把关键帧转为结构化场景描述 JSON（对象/类别/尺寸/位置/遮挡）
- 跨帧 ID 漂移、尺寸漂移 >30%、schema 校验失败、多人同框等问题的处理

## 输入输出约定
- 输入：`data/analysis/<clip_id>/frames_key/`（来自 preprocess）
- 输出：`semantic/fNNNNNN.json`（逐帧）、`asset_store.sqlite`（资产库）、
  `scene.json`（融合场景）、`flags.json`（multi_subject / needs_reparse）

## 步骤
1. 确认环境变量 `ANTHROPIC_API_KEY` 已设置
2. 运行解析（默认 claude-opus-5-5；429/5xx 自动退避，4xx 自动降级 sonnet）：
   ```powershell
   python -m src.vision.semantic_parse --clip <clip_id>
   # 可选：只解析指定帧 python -m src.vision.semantic_parse --clip <id> --frames 0,24,48
   ```
3. 离线批量（成本减半，例行帧；无前帧上下文链）：
   ```powershell
   python -m src.vision.batch_parse --clip <id> submit   # 提交
   python -m src.vision.batch_parse --clip <id> status   # 轮询
   python -m src.vision.batch_parse --clip <id> fetch    # 回收入库
   ```
4. 检查 `scene.json`：资产数量、needs_reparse 列表、类别合理性
5. `flags.json.multi_subject=true` → 整对打回（需求禁止多主体，风险 #3）

## 引用标准
《白模实时建模工作流设计文档》§4.3（schema 与一致性流程）、§7.1（模型路由）、§7.3（集成要点：流式/缓存/Batch/重试）

## 失败处理
- 单帧 JSON 解析失败 → 程序自动带错误提示重试 1 次；仍失败记录 semantic_failures
- 全部帧失败 → 检查 API Key / 网络 / 模型名，再 `--force` 重跑 semantic
- 跨帧 ID 漂移（同一棵树两个 id）→ 用 `src.vision.asset_store.SceneAssetStore.mark_merged(keep, drop)` 合并后重新 export
- 尺寸漂移 needs_reparse → 对应帧单独重解析：`--frames <frame>`，人工确认后清除 flags
