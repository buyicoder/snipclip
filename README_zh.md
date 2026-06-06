# SnipClip 🎬

> AI 驱动的全自动视频剪辑 — 引擎动手，Claude 动脑。

SnipClip 是一个两层架构的自动视频剪辑框架：

- **SnipClip Engine**：Python CLI 工具包，提供视频处理的基础能力 —— 音频提取、Whisper 语音转写、剪切/拼接、字幕生成。零智能，纯执行。
- **SnipClip Skill**：Claude Code 技能，充当"AI 剪辑导演"。阅读转写文本，理解你的意图，做出所有剪辑决策。

## 快速开始

### 1. 安装

```bash
pip install snipclip
snipclip setup          # 自动下载 FFmpeg（如需要）
```

### 2. 配合 Claude Code 使用

将 skill 文件复制到 Claude Code 的 skills 目录：

```bash
cp skill/snipclip.md ~/.claude/skills/snipclip.md
```

然后在 Claude Code 中：

```
/snipclip
把这段会议视频剪成 3 分钟，只保留结论和决策。
```

Claude 会自动：
1. 探测视频信息
2. 转写语音为文字
3. 分析内容，决定保留/删除
4. 调用引擎执行剪切
5. 交付成品视频

### 3. 独立 CLI 使用

```bash
# 查看视频信息
snipclip probe video.mp4

# 语音转文字
snipclip transcribe video.mp4 -o transcript.json

# 按时间段剪切（保留模式）
snipclip cut video.mp4 --keep segments.json -o output.mp4

# 生成字幕
snipclip subtitle video.mp4 transcript.json
```

## 剪辑人格

Claude Code Skill 内置了预设的剪辑策略：

| 人格 | 适用场景 | 剪辑逻辑 |
|------|---------|---------|
| 🎓 教程 | 教程、课程 | 保留知识点和操作步骤，删除闲聊和重复 |
| 💼 会议 | 会议、访谈 | 保留结论、决策、行动项，删除讨论过程 |
| ⚡ 短视频 | 抖音、Reels | 快节奏跳切，每段不超过 30 秒 |
| 🎙️ 播客 | Vlog、播客 | 去静音、去废话，保留叙事流畅感 |
| 🎯 自定义 | 任意场景 | 完全按你的文字描述执行 |

## 架构

```
Claude Code (AI 导演)             SnipClip Engine (执行器)
┌──────────────────────┐        ┌──────────────────────┐
│ • 理解用户意图        │───────▶│ probe   → 视频信息   │
│ • 分析转写文本        │        │ extract → 音频提取  │
│ • 决定保留/删除       │        │ transcribe → 转写   │
│ • 展示剪辑方案        │        │ cut     → 输出视频  │
│ • 接受反馈迭代        │        │ subtitle → 字幕     │
└──────────────────────┘        └──────────────────────┘
```

## 环境要求

- Python 3.10+
- FFmpeg（可通过 `snipclip setup` 自动下载）
- Claude Code（用于 AI 智能剪辑）
- CPU：任何机器都能跑
- GPU：自动检测（CUDA），加速转写

## 开发

```bash
git clone https://github.com/your-username/snipclip.git
cd snipclip
pip install -e ".[dev]"
python scripts/make_fixtures.py   # 生成测试视频
pytest                              # 运行测试
```

## License

MIT
