---
name: camera-solve
description: 相机轨迹解算（MASt3R/GLOMAP）、米制尺度恢复、地面拟合与世界坐标融合。当用户要求解算相机、修复白模视角漂移、处理解算失败降级或资产落位异常时使用。
---

# camera-solve —— 相机与几何解算（Stage 4）

## 触发条件
- 需要相机轨迹 + 稀疏点云（camera.json）与场景米制尺度
- 白模视角漂移 / 资产悬空或入地 / 尺度明显错误

## 输入输出约定
- 输入：`frames_full/`（SfM 序列）、`scene.json`（语义尺寸锚点）
- 输出：`camera.json`（全帧位姿，convention=blender）、
  `sfm/`（COLMAP 文本 + 点云）、`scene.json` 增补 world_position / trajectory_world、
  `poses.json`（世界 3D 关节）

## 步骤
1. 配置后端（二选一，双路互备）：
   - MASt3R（优先）：clone 到 `external/mast3r`，config `camera.mast3r_cmd` 填命令模板
     （须支持 `{frames_dir}` 与 `{out_dir}` 占位，输出 COLMAP 文本格式）
   - GLOMAP（备选）：同理配置 `camera.glomap_cmd`
2. 运行解算与内插：
   ```powershell
   python -m src.sfm.camera_solve --clip <clip_id>
   ```
3. 尺度/地面/落位/姿态提升：
   ```powershell
   python -m src.sfm.scale_ground --clip <clip_id>
   ```
4. 校验 `scene.json`：人物 world_position 是否在相机前方、trajectory 是否贴合运动路线

## 引用标准
《白模实时建模工作流设计文档》§4.4（解算步骤与失败降级）、§9 风险 #1/#2

## 失败处理
- 后端未配置/失败 → 自动降级恒定机位（confidence=0.1，质检加严）；不阻塞流水线
- 快速旋转/运动模糊失败 → 减小 `camera.keyframe_subsample` 重试（关键帧稀疏化 + 位姿内插）
- 地面拟合差（ground_method=default_z0）→ 点云不足；可只跑 `--stages world` 反复调参
- 尺度异常 → 检查语义人物 size_est_m 是否落在 1.5–2.1m；无锚点时 scale=1（低置信）
- 视角完全不可用 → 以 Claude 语义视角序列兜底：手写 camera.json（低置信度标注）
