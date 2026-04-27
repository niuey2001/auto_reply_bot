from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"
MAIN_SCRIPT_PATH = BASE_DIR / "src" / "main.py"

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
)


class BotProcessManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._log_buffer: deque[str] = deque(maxlen=300)
        self._log_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._config_path = str(DEFAULT_CONFIG_PATH)

    def _append_log(self, line: str) -> None:
        text = line.rstrip("\n")
        if not text:
            return
        ts = time.strftime("%H:%M:%S")
        self._log_buffer.append(f"[{ts}] {text}")

    def _stream_output(self, proc: subprocess.Popen[str]) -> None:
        if not proc.stdout:
            return
        try:
            for line in proc.stdout:
                self._append_log(line)
        finally:
            code = proc.poll()
            self._append_log(f"[system] bot exited with code={code}")

    def start(self, config_path: str) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "bot is already running"
            if not os.path.exists(config_path):
                return False, f"config file not found: {config_path}"

            cmd = [sys.executable, "-X", "utf8", "-u", str(MAIN_SCRIPT_PATH), "--config", config_path]
            self._append_log(f"[system] starting: {' '.join(cmd)}")
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
            )
            self._started_at = time.time()
            self._config_path = config_path
            self._log_thread = threading.Thread(
                target=self._stream_output,
                args=(self._proc,),
                daemon=True,
            )
            self._log_thread.start()
            return True, "bot started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self.is_running():
                return False, "bot is not running"
            assert self._proc is not None
            self._append_log("[system] stopping bot process...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._append_log("[system] terminate timeout, force kill")
                self._proc.kill()
                self._proc.wait(timeout=5)
            return True, "bot stopped"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, object]:
        running = self.is_running()
        pid = self._proc.pid if running and self._proc else None
        uptime_sec = int(time.time() - self._started_at) if running else 0
        return {
            "running": running,
            "pid": pid,
            "uptime_sec": uptime_sec,
            "config_path": self._config_path,
            "logs": list(self._log_buffer),
        }


manager = BotProcessManager()


FORM_FIELDS: list[dict[str, str]] = [
    {"key": "runtime.poll_interval_sec", "label": "轮询间隔(秒)", "hint": "每次 OCR 检测的时间间隔，越小越及时。", "type": "number"},
    {"key": "runtime.min_text_chars", "label": "最短消息长度", "hint": "低于该字符数的识别结果将忽略，避免误触发。", "type": "number"},
    {"key": "runtime.post_send_cooldown_sec", "label": "发送后冷却(秒)", "hint": "发送后暂停检测，减少 OCR 延迟导致的重复回复。", "type": "number"},
    {"key": "runtime.real_send", "label": "真实发送", "hint": "关闭后仅打印拟发送内容（推荐调试阶段关闭）。", "type": "checkbox"},
    {"key": "runtime.enable_llm", "label": "启用 LLM", "hint": "关闭后不调用模型，仅输出识别到的消息。", "type": "checkbox"},
    {"key": "runtime.debug_llm", "label": "调试日志", "hint": "开启后打印模型输入输出，方便排查。", "type": "checkbox"},
    {"key": "runtime.self_bubble_side", "label": "自己消息气泡方向", "hint": "通常为 right（微信常见右侧是自己消息）。", "type": "text"},
    {"key": "runtime.side_split_x_ratio", "label": "左右分割比例", "hint": "按聊天区宽度分割左右消息，常用 0.6。", "type": "number"},
    {"key": "runtime.partner_bottom_window_ratio", "label": "对方消息底部窗口比例", "hint": "仅在底部区域找最新对方消息，减少历史消息干扰。", "type": "number"},
    {"key": "screen.chat_region.left", "label": "聊天区 left", "hint": "聊天截图区域左上角 X 坐标。", "type": "number"},
    {"key": "screen.chat_region.top", "label": "聊天区 top", "hint": "聊天截图区域左上角 Y 坐标。", "type": "number"},
    {"key": "screen.chat_region.width", "label": "聊天区 width", "hint": "聊天截图区域宽度。", "type": "number"},
    {"key": "screen.chat_region.height", "label": "聊天区 height", "hint": "聊天截图区域高度。", "type": "number"},
    {"key": "screen.input_box_point.x", "label": "输入框 X", "hint": "点击输入框时的屏幕 X 坐标。", "type": "number"},
    {"key": "screen.input_box_point.y", "label": "输入框 Y", "hint": "点击输入框时的屏幕 Y 坐标。", "type": "number"},
    {"key": "llm.api_base", "label": "API Base", "hint": "模型服务地址，如 https://api.openai.com/v1。", "type": "text"},
    {"key": "llm.api_key", "label": "API Key", "hint": "直接填写 API Key 值，不再依赖环境变量。", "type": "password"},
    {"key": "llm.model", "label": "模型名", "hint": "例如 gpt-4.1-mini。", "type": "text"},
    {"key": "llm.temperature", "label": "temperature", "hint": "回复随机性，0 更稳定，越大越发散。", "type": "number"},
    {"key": "llm.max_output_tokens", "label": "max_output_tokens", "hint": "模型最大输出 token 数。", "type": "number"},
    {"key": "llm.system_prompt", "label": "系统提示词(system_prompt)", "hint": "用于约束回复风格与规则。", "type": "textarea"},
]


def _read_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("invalid config structure: root must be mapping")
    return data


def _write_yaml_config(config_path: str, data: dict) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _get_nested(data: dict, key: str):
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(data: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/config")
def get_config() -> object:
    config_path = request.args.get("path", str(DEFAULT_CONFIG_PATH))
    if not os.path.exists(config_path):
        return jsonify({"ok": False, "error": f"config file not found: {config_path}"}), 404
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"ok": True, "path": config_path, "content": content})


@app.post("/api/config")
def save_config() -> object:
    data = request.get_json(silent=True) or {}
    config_path = str(data.get("path") or DEFAULT_CONFIG_PATH)
    content = data.get("content")
    if not isinstance(content, str):
        return jsonify({"ok": False, "error": "content is required"}), 400
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return jsonify({"ok": True, "path": config_path})


@app.get("/api/config/form")
def get_config_form() -> object:
    config_path = request.args.get("path", str(DEFAULT_CONFIG_PATH))
    if not os.path.exists(config_path):
        return jsonify({"ok": False, "error": f"config file not found: {config_path}"}), 404
    try:
        data = _read_yaml_config(config_path)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"failed to parse yaml: {exc}"}), 400
    values = {item["key"]: _get_nested(data, item["key"]) for item in FORM_FIELDS}
    return jsonify({"ok": True, "path": config_path, "fields": FORM_FIELDS, "values": values})


@app.post("/api/config/form")
def save_config_form() -> object:
    payload = request.get_json(silent=True) or {}
    config_path = str(payload.get("path") or DEFAULT_CONFIG_PATH)
    updates = payload.get("values")
    if not isinstance(updates, dict):
        return jsonify({"ok": False, "error": "values is required"}), 400
    if not os.path.exists(config_path):
        return jsonify({"ok": False, "error": f"config file not found: {config_path}"}), 404
    try:
        data = _read_yaml_config(config_path)
        editable_keys = {item["key"] for item in FORM_FIELDS}
        for key, value in updates.items():
            if key not in editable_keys:
                continue
            _set_nested(data, key, value)
        _write_yaml_config(config_path, data)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"failed to save yaml: {exc}"}), 400
    return jsonify({"ok": True, "path": config_path})


@app.get("/api/bot/status")
def bot_status() -> object:
    return jsonify({"ok": True, "data": manager.status()})


@app.post("/api/bot/start")
def bot_start() -> object:
    data = request.get_json(silent=True) or {}
    config_path = str(data.get("config_path") or DEFAULT_CONFIG_PATH)
    ok, message = manager.start(config_path)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": message, "data": manager.status()}), code


@app.post("/api/bot/stop")
def bot_stop() -> object:
    ok, message = manager.stop()
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": message, "data": manager.status()}), code


def main() -> None:
    parser = argparse.ArgumentParser(description="Web dashboard for auto reply bot")
    parser.add_argument("--host", default="127.0.0.1", help="host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="port (default: 7860)")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
