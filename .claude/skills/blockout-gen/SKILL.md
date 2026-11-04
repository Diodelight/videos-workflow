---
name: blockout-gen
description: Blender 白模生成与 EEVEE 渲染（图元实例化、相机动画、人物姿态/步态、碰撞体、材质、合规校验）。当用户要求生成白模、修正白模资产、调整材质/碰撞/渲染参数时使用。
---

# blockout-gen —— Blender 白模生成（Stage 5）

## 触发条件
- 依据 scene.json 生成白模并渲染序列
- 白模不合规（精细网格/纹理/缺门洞等）、需要资产修正、材质/碰撞调整

## 输入输出约定
- 输入：`scene.json`（含 world_position/trajectory_world）、`camera.json`、
  `meta.json`、可选 `poses.json`
- 输出：`render/white_NNNN.png` 序列、`blockout.blend`、`compliance.json`

## 步骤
1. 流水线调用（推荐，自动传参）：
   ```powershell
   python -m src.blender_gen.generate --clip <clip_id>
   # 或 python run_pipeline.py --clip <id> --stages generate
   ```
2. 手动 headless（等效）：
   ```powershell
   blender -b -P src/blender_gen/gen_blockout.py -- --scene <scene.json> --camera <camera.json> --meta <meta.json> --poses <poses.json|none> --out <render目录> --blend <blend路径> --compliance <报告路径> --background green
   ```
3. 人工修正（调试期）：启动 blender-mcp（Blender 内启用 addon 后 `uvx blender-mcp`），
   打开 `blockout.blend` 交互调整；改完重跑 compose/qa
4. 核对 `compliance.json` 全部 PASS（§4.5 Checklist：无精细网格、无纹理、
   人物头/躯干/四肢、树干/冠、建筑门洞或通道、碰撞体存在、无特效、面数 ≤5 万）

## 引用标准
《白模实时建模工作流设计文档》§2.2（几何标准）、§4.5（生成步骤 1–8）、§8（面数指标）

## 失败处理
- 找不到 Blender → config.yaml `blender.executable` 填绝对路径
- 人物动作僵硬 → 有 poses.json 用骨骼关键帧；无姿态数据自动 walk_cycle 降级步态
- 资产未落位（world_position=null）→ 回查 world 阶段（相机/地面），`--stages world,generate --force`
- 渲染过慢 → 降低 `blender.eevee_samples`；白模为简单几何，正常应远快于实时
- 合规 FAIL → 按 compliance.json 各项逐条修正 scene.json 或生成参数后重跑
