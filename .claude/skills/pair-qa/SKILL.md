---
name: pair-qa
description: 成对数据质检与交付（对齐/运动同步/纯净度/白模合规/音频自动检查、30% 人工抽检、打回返工流程）。当用户要求质检成对数据、分析 qa_report、处理打回返工时使用。
---

# pair-qa —— 成对质检与交付（Stage 6–7）

## 触发条件
- 白模视频产出后的自动质检
- 分析 qa_report.json、抽检 side_by_side、批量交付前验收、打回返工

## 输入输出约定
- 输入：`data/analysis/<clip_id>/白模视频.mp4`、meta.json、compliance.json、purity.json
- 输出：`qa_report.json`；交付目录 `data/output/<clip_id>/`
  （白模视频.mp4、目标视频.mp4、对齐映射说明.json、side_by_side.mp4）

## 步骤
1. 合成交付（若未跑）：`python run_pipeline.py --clip <id> --stages compose,qa`
2. 自动质检（对齐 0 错帧 / 运动相关系数 / 合规 / 纯净度 / 音频）：
   ```powershell
   python -m src.qa.qa_check --clip <clip_id>
   ```
3. 读 `qa_report.json` 逐项核对；失败项 → 对应返工路径（见下）
4. 人工抽检：按批次 30% 随机抽 delivery 目录，看 side_by_side.mp4
   （主体/场景/运镜/时序对应、轮廓可辨、无 UI）；抽检准确率 ≥95%，未达标整批返工
5. 批量统计：遍历 `data/output/*` 汇总各 clip 的 qa_report 通过率

## 引用标准
《白模实时建模工作流设计文档》§4.6、§4.7（检查表与验收策略）、§8（指标）

## 失败处理
- alignment FAIL → meta 与渲染帧数不一致：查 generate 日志；`--stages generate,compose,qa --force`
- motion_sync FAIL → 相机轨迹漂移或人物错位：回 camera-solve；仍不行按低置信度标注
- blockout_compliance FAIL → 回 blockout-gen 修正
- audio FAIL → 检查 audio.m4a 与 compose 的 -map 参数
- 多人同框打回（flags.json.multi_subject）→ 更换片段素材，不返工生成
- 返工后重跑：`python run_pipeline.py --clip <id> --force --stages <需重跑的阶段>`
