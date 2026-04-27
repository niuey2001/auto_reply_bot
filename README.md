# 聊天软件自动回复工具（屏幕扫描 + OCR + 大模型）

该工具会循环截取你指定的聊天窗口区域，做 OCR 识别后调用大模型生成回复，并自动点击输入框发送。

## 1) 安装

```powershell
cd auto_reply_bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 配置

```powershell
copy config.example.yaml config.yaml
```

编辑 `config.yaml` 重点修改：

- `screen.chat_region`: 聊天内容区域坐标
- `screen.input_box_point`: 输入框可点击坐标
- `runtime.self_bubble_side`: 自己消息气泡方向（常见是 `right`）
- `runtime.side_split_x_ratio`: 左右分割点（按聊天区域宽度比例）
- `runtime.post_send_cooldown_sec`: 发送后冷却秒数，防止 OCR 延迟引发二次触发
- `runtime.partner_bottom_window_ratio`: 仅在聊天底部区域挑选对方最新消息
- `runtime.self_echo_guard_sec`: 发送后短时间内忽略与自己刚发内容一致的“对方消息”
- `runtime.real_send`: 是否真实发送。`false` 时只在控制台打印拟发送内容，不点输入框、不发送
- `runtime.enable_llm`: 是否调用大模型。`false` 时不调用 LLM、不发送，仅打印识别到的最新消息/上下文（用于测试阶段避免浪费 token）
- `runtime.post_session_click_delay_sec` / `red_badge_min_pixels`: 左侧未读红点检测相关（见下）
- `screen.session_list_region`（可选）: 左侧会话列表的截图区域。填写后**仅当**该区域内检测到未读红点时才处理：先自动点击红点再 OCR 主聊天区；不填则像以前一样持续轮询 `chat_region`
- `runtime.debug_llm`: 打印模型调试日志（输入/输出，自动截断）
- `llm.model`: 使用的大模型名称
- `llm.api_base`: OpenAI 或兼容网关地址
- `llm.api_key`: 直接填写 API Key（推荐）
- `llm.api_key_env`: 兼容旧方式，API Key 环境变量名（可选）

推荐直接在 `config.yaml` 中填写：

```yaml
llm:
  api_key: "你的密钥"
```

旧方式（环境变量）仍兼容：

```powershell
$env:OPENAI_API_KEY="你的密钥"
```

如果使用 NVIDIA 网关（`https://integrate.api.nvidia.com/v1`），建议改成：

```powershell
$env:NVIDIA_API_KEY="nvapi-你的密钥"
```

## 3) 运行

```powershell
python src/main.py --config config.yaml
```

运行后将持续轮询。把鼠标快速移动到屏幕左上角可触发 failsafe 退出。

## 4) 前端控制台（新增）

提供了一个本地网页控制台，可编辑配置并启动/停止机器人。

```powershell
python src/web_app.py --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860` 即可使用。页面支持：

- 通过表单加载 / 保存主要配置参数
- 查看机器人运行状态（PID、运行时长）
- 启动 / 停止机器人
- 查看最近运行日志

## 5) 如何校准坐标

- 先打开目标聊天软件窗口并固定位置
- 使用截图工具或开发者工具获取聊天区矩形（left/top/width/height）
- 用鼠标指向输入框，记录坐标后填入 `input_box_point`

## 6) 重要提醒

- 本项目是原型，请仅在你授权的账号和场景使用
- OCR 可能误识别，建议先小范围测试
- 自动发送有风险，建议先把 `real_send` 设为 `false` 仅做控制台演练
