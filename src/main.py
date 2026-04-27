from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pyautogui
import pyperclip
import requests
import yaml
import numpy as np
from mss import mss
from rapidocr_onnxruntime import RapidOCR


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass
class AppConfig:
    poll_interval_sec: float
    min_text_chars: int
    unchanged_skip_sec: float
    post_send_cooldown_sec: float
    real_send: bool
    enable_llm: bool
    post_session_click_delay_sec: float
    session_list_region: Region | None
    red_badge_min_pixels: int
    chat_region: Region
    input_box_point: tuple[int, int]
    model: str
    system_prompt: str
    temperature: float
    max_output_tokens: int
    api_base: str
    api_key: str
    api_key_env: str
    api_style: str
    self_bubble_side: str
    side_split_x_ratio: float
    debug_llm: bool
    partner_bottom_window_ratio: float
    self_echo_guard_sec: float


@dataclass
class OCRLine:
    text: str
    center_x: float
    center_y: float
    min_x: float
    max_x: float


@dataclass
class SessionClickState:
    last_click_pos: tuple[int, int] | None = None
    last_click_time: float = 0.0
    # When we suspect a false-positive red detection, extend cooldown.
    cooldown_until: float = 0.0
    # Temporarily ignored click positions: (x, y, until_ts)
    blocked_positions: list[tuple[int, int, float]] | None = None


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sl_raw = (data.get("screen") or {}).get("session_list_region")
    session_list_region: Region | None
    if sl_raw:
        session_list_region = Region.from_dict(sl_raw)
    else:
        session_list_region = None

    return AppConfig(
        poll_interval_sec=float(data["runtime"]["poll_interval_sec"]),
        min_text_chars=int(data["runtime"]["min_text_chars"]),
        unchanged_skip_sec=float(data["runtime"]["unchanged_skip_sec"]),
        post_send_cooldown_sec=float(data["runtime"].get("post_send_cooldown_sec", 3)),
        real_send=bool(data["runtime"].get("real_send", True)),
        enable_llm=bool(data["runtime"].get("enable_llm", True)),
        post_session_click_delay_sec=float(data["runtime"].get("post_session_click_delay_sec", 0.45)),
        session_list_region=session_list_region,
        red_badge_min_pixels=int(data["runtime"].get("red_badge_min_pixels", 5)),
        chat_region=Region.from_dict(data["screen"]["chat_region"]),
        input_box_point=(
            int(data["screen"]["input_box_point"]["x"]),
            int(data["screen"]["input_box_point"]["y"]),
        ),
        model=str(data["llm"]["model"]),
        system_prompt=str(data["llm"]["system_prompt"]),
        temperature=float(data["llm"]["temperature"]),
        max_output_tokens=int(data["llm"]["max_output_tokens"]),
        api_base=str(data["llm"]["api_base"]).rstrip("/"),
        api_key=str((data.get("llm") or {}).get("api_key", "") or ""),
        api_key_env=str((data.get("llm") or {}).get("api_key_env", "") or ""),
        api_style=str(data["llm"].get("api_style", "chat_completions")).strip().lower(),
        self_bubble_side=str(data["runtime"].get("self_bubble_side", "right")).strip().lower(),
        side_split_x_ratio=float(data["runtime"].get("side_split_x_ratio", 0.60)),
        debug_llm=bool(data["runtime"].get("debug_llm", False)),
        partner_bottom_window_ratio=float(data["runtime"].get("partner_bottom_window_ratio", 0.45)),
        self_echo_guard_sec=float(data["runtime"].get("self_echo_guard_sec", 12)),
    )


def ocr_lines(ocr_engine: RapidOCR, region: Region) -> list[OCRLine]:
    with mss() as sct:
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        shot = sct.grab(monitor)
        # RapidOCR expects an image array/path, not raw RGB bytes.
        img = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        bgr_img = img[:, :, :3]
        result, _ = ocr_engine(bgr_img, use_det=True, use_cls=True, use_rec=True)

    if not result:
        return []

    lines: list[OCRLine] = []
    for item in result:
        if len(item) < 2:
            continue
        box = item[0]
        text = str(item[1]).strip()
        if not text:
            continue

        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            center_x = float(sum(xs) / len(xs))
            center_y = float(sum(ys) / len(ys))
            min_x = float(min(xs))
            max_x = float(max(xs))
        except Exception:
            # Fallback for unexpected OCR box formats.
            center_x = region.width / 2.0
            center_y = 0.0
            min_x = center_x
            max_x = center_x
        lines.append(
            OCRLine(
                text=text,
                center_x=center_x,
                center_y=center_y,
                min_x=min_x,
                max_x=max_x,
            )
        )
    return lines


def build_chat_context(lines: list[OCRLine]) -> str:
    ordered = sorted(lines, key=lambda x: (x.center_y, x.center_x))
    return "\n".join(line.text for line in ordered)


def capture_region_bgr(region: Region) -> np.ndarray:
    with mss() as sct:
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        shot = sct.grab(monitor)
        img = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return img[:, :, :3].copy()


def _is_position_blocked(state: SessionClickState, pos: tuple[int, int], now: float, radius: int = 8) -> bool:
    if not state.blocked_positions:
        return False
    still_valid: list[tuple[int, int, float]] = []
    blocked = False
    r2 = radius * radius
    for bx, by, until_ts in state.blocked_positions:
        if until_ts <= now:
            continue
        still_valid.append((bx, by, until_ts))
        dx = bx - pos[0]
        dy = by - pos[1]
        if (dx * dx + dy * dy) <= r2:
            blocked = True
    state.blocked_positions = still_valid
    return blocked


def _block_position(state: SessionClickState, pos: tuple[int, int], now: float, seconds: float) -> None:
    until_ts = now + max(0.0, seconds)
    if state.blocked_positions is None:
        state.blocked_positions = []
    state.blocked_positions.append((pos[0], pos[1], until_ts))


def find_unread_red_badge_candidates(
    bgr: np.ndarray,
    min_pixels: int,
    *,
    max_pixels: int = 1200,
    max_box_size: int = 44,
    prefer_right_ratio: float = 0.70,
    right_zone_min_ratio: float = 0.45,
    debug: bool = False,
) -> list[tuple[int, int]]:
    """
    Find an unread-style red badge center by connected components.

    Rationale:
    - Avatars often contain red pixels; picking the topmost red pixel is unstable.
    - Unread badges are usually small, near-circular blobs with limited box size.
    """
    h, w = bgr.shape[:2]
    b = bgr[:, :, 0].astype(np.int32)
    g = bgr[:, :, 1].astype(np.int32)
    r = bgr[:, :, 2].astype(np.int32)

    # Conservative "red dot" mask (works for WeChat-style unread badges).
    # Slightly relaxed to catch darker "unread count" bubbles too.
    mask = (r > 165) & (r > g + 35) & (r > b + 35) & (g < 150) & (b < 150)
    if not mask.any():
        return []

    visited = np.zeros((h, w), dtype=bool)
    ys_all, xs_all = np.where(mask)

    # Iterate pixels in scanline order so we naturally find topmost components first.
    order = np.argsort(ys_all * w + xs_all)
    scored: list[tuple[tuple[float, float, float, float], tuple[int, int]]] = []

    for idx in order:
        y0 = int(ys_all[idx])
        x0 = int(xs_all[idx])
        if visited[y0, x0]:
            continue

        # Flood fill component.
        stack = [(y0, x0)]
        visited[y0, x0] = True
        area = 0
        sum_x = 0
        sum_y = 0
        min_x = x0
        max_x = x0
        min_y = y0
        max_y = y0

        while stack:
            y, x = stack.pop()
            area += 1
            sum_x += x
            sum_y += y
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            # 8-neighborhood for better circle connectivity.
            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx]:
                        continue
                    if not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            # Early abort huge blobs.
            if area > max_pixels:
                break

        if area < min_pixels or area > max_pixels:
            continue

        box_w = (max_x - min_x + 1)
        box_h = (max_y - min_y + 1)
        if box_w > max_box_size or box_h > max_box_size:
            continue

        # Rough circularity: bounding box aspect ratio close to 1.
        aspect = box_w / max(1.0, float(box_h))
        if aspect < 0.55 or aspect > 1.8:
            continue

        cx = int(round(sum_x / max(1.0, float(area))))
        cy = int(round(sum_y / max(1.0, float(area))))

        if debug:
            print(
                f"[session][cand] area={area} box=({box_w}x{box_h}) center=({cx},{cy}) "
                f"bbox=({min_x},{min_y})-({max_x},{max_y})"
            )

        # Heuristic scoring:
        # - strongly prefer right-side blobs (badge is usually on avatar right edge)
        # - then prefer smaller y (topmost among right-side candidates)
        # - then prefer larger x
        # - then medium sizes (avoid avatar red patches)
        right_penalty = 0.0 if cx >= int(w * right_zone_min_ratio) else 1.0
        x_rank = -float(cx)  # larger cx should win among same row
        # Encourage "near-right" components when there are multiple options.
        if cx < int(w * prefer_right_ratio):
            x_rank += 1000.0
        size_penalty = abs(area - max(min_pixels * 3, 30)) / 200.0
        key = (right_penalty, float(cy), x_rank, size_penalty)
        scored.append((key, (cx, cy)))

    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored]


def find_unread_red_badge_center(
    bgr: np.ndarray,
    min_pixels: int,
    *,
    max_pixels: int = 1200,
    max_box_size: int = 44,
    prefer_right_ratio: float = 0.70,
    right_zone_min_ratio: float = 0.45,
    debug: bool = False,
) -> tuple[int, int] | None:
    candidates = find_unread_red_badge_candidates(
        bgr,
        min_pixels,
        max_pixels=max_pixels,
        max_box_size=max_box_size,
        prefer_right_ratio=prefer_right_ratio,
        right_zone_min_ratio=right_zone_min_ratio,
        debug=debug,
    )
    if not candidates:
        return None
    return candidates[0]


def try_click_unread_badge(cfg: AppConfig, state: SessionClickState, now: float) -> bool:
    """
    If session_list_region is set: scan for a red badge; click it and return True.
    If not configured: return True so the main loop can keep legacy always-on behavior.
    If configured but no badge: return False (skip this round).
    """
    if cfg.session_list_region is None:
        return True
    if now < state.cooldown_until:
        return False
    bgr = capture_region_bgr(cfg.session_list_region)
    candidates = find_unread_red_badge_candidates(bgr, cfg.red_badge_min_pixels, debug=cfg.debug_llm)
    if not candidates:
        if cfg.debug_llm:
            print("[session] no unread red badge in session_list_region; skip")
        return False
    chosen: tuple[int, int] | None = None
    for cx, cy in candidates:
        cand_pos = (cfg.session_list_region.left + cx, cfg.session_list_region.top + cy)
        if _is_position_blocked(state, cand_pos, now):
            continue
        chosen = cand_pos
        break
    if chosen is None:
        if cfg.debug_llm:
            print("[session] all detected badges are temporarily blocked; skip")
        return False
    pos = chosen
    abs_x, abs_y = pos
    # Debounce: if we're clicking the exact same spot repeatedly, slow down.
    if state.last_click_pos == pos and (now - state.last_click_time) < max(0.8, cfg.poll_interval_sec * 2):
        if cfg.debug_llm:
            print(f"[session] same badge pos {pos} within debounce window; skip this round")
        return False
    if cfg.debug_llm:
        print(f"[session] click unread badge at screen=({abs_x}, {abs_y})")
    pyautogui.click(abs_x, abs_y)
    time.sleep(max(0.0, cfg.post_session_click_delay_sec))
    # Re-check once: if the red badge is still detected at roughly the same place,
    # it's likely a false-positive (or click didn't change state). Extend cooldown to avoid spamming.
    try:
        bgr2 = capture_region_bgr(cfg.session_list_region)
        center2 = find_unread_red_badge_center(bgr2, cfg.red_badge_min_pixels)
        if center2 is not None:
            abs2 = (cfg.session_list_region.left + center2[0], cfg.session_list_region.top + center2[1])
            if abs2 == pos:
                state.cooldown_until = now + max(2.0, cfg.poll_interval_sec * 5)
                _block_position(state, pos, now, max(20.0, cfg.poll_interval_sec * 20))
                if cfg.debug_llm:
                    print(f"[session] badge still at {pos} after click; cooldown until {state.cooldown_until:.2f}")
    except Exception:
        # Never block the main loop due to re-check errors.
        pass
    state.last_click_pos = pos
    state.last_click_time = now
    return True


def latest_partner_message(lines: list[OCRLine], cfg: AppConfig, region: Region) -> str:
    if cfg.self_bubble_side not in {"left", "right"}:
        raise RuntimeError("runtime.self_bubble_side must be 'left' or 'right'")

    split_x = region.width * cfg.side_split_x_ratio
    bottom_min_y = region.height * max(0.0, min(1.0, 1.0 - cfg.partner_bottom_window_ratio))
    candidates: list[OCRLine] = []
    for line in lines:
        if line.center_y < bottom_min_y:
            continue
        # Use text start position (left boundary) instead of center_x for more stable side detection.
        start_x = line.min_x
        if cfg.self_bubble_side == "right":
            is_self = start_x >= split_x
            is_partner = start_x < split_x
        else:
            is_self = start_x < split_x
            is_partner = start_x >= split_x

        if is_self:
            continue
        if not is_partner:
            continue
        if not is_self:
            candidates.append(line)

    if not candidates:
        return ""
    return max(candidates, key=lambda x: x.center_y).text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return "".join(text.strip().lower().split())


def is_echo_reply(reply: str, partner_latest_text: str) -> bool:
    a = normalize_text(reply)
    b = normalize_text(partner_latest_text)
    if not a or not b:
        return False
    if a == b:
        return True
    # Guard near-duplicate outputs like added punctuation/particles.
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.9


def preview_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def seems_non_chat_capture(chat_text: str) -> bool:
    markers = [
        "config.yaml",
        "readme",
        "runtime.",
        "python src/main.py",
        "debug_llm",
        "api_style",
    ]
    lowered = chat_text.lower()
    hit = sum(1 for m in markers if m in lowered)
    return hit >= 2


def call_llm(cfg: AppConfig, user_text: str) -> str:
    api_key = (cfg.api_key or "").strip()
    if not api_key and cfg.api_key_env:
        api_key = os.getenv(cfg.api_key_env, "").strip()
    if (api_key.startswith('"') and api_key.endswith('"')) or (
        api_key.startswith("'") and api_key.endswith("'")
    ):
        api_key = api_key[1:-1].strip()
    if not api_key:
        raise RuntimeError(
            "missing API key. set llm.api_key in config.yaml "
            "(or fallback to llm.api_key_env environment variable)"
        )
    if "nvidia.com" in cfg.api_base and not api_key.startswith("nvapi-"):
        raise RuntimeError(
            f"invalid NVIDIA API key format in env var: {cfg.api_key_env}. "
            "Expected key to start with 'nvapi-'."
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if cfg.api_style == "responses":
        url = f"{cfg.api_base}/responses"
        payload = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": cfg.system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
        }
    else:
        # OpenAI-compatible gateways (including many third-party providers) usually expose this endpoint.
        url = f"{cfg.api_base}/chat/completions"
        payload = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_output_tokens,
            "messages": [
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
    if cfg.debug_llm:
        print(f"[llm] request model={cfg.model} api_style={cfg.api_style}")
        print(f"[llm] request input:\n{preview_text(user_text)}")

    resp = requests.post(url, headers=headers, json=payload, timeout=45)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text.strip()
        if len(detail) > 300:
            detail = detail[:300] + "..."
        raise RuntimeError(
            f"http {resp.status_code} calling {url}. detail: {detail or 'no response body'}"
        ) from exc
    data = resp.json()

    if cfg.api_style == "responses":
        if "output_text" in data and data["output_text"]:
            output_text = str(data["output_text"]).strip()
            if cfg.debug_llm:
                print(f"[llm] response output:\n{preview_text(output_text)}")
            return output_text

        output = data.get("output", [])
        texts: list[str] = []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
        output_text = "\n".join(texts).strip()
        if cfg.debug_llm:
            print(f"[llm] response output:\n{preview_text(output_text)}")
        return output_text

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            output_text = content.strip()
            if cfg.debug_llm:
                print(f"[llm] response output:\n{preview_text(output_text)}")
            return output_text
    if cfg.debug_llm:
        print("[llm] response output is empty.")
    return ""


def send_message(text: str, input_point: tuple[int, int]) -> None:
    pyautogui.click(*input_point)
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    pyautogui.press("enter")


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    ocr_engine = RapidOCR()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1

    last_seen_hash = ""
    suppress_until = 0.0
    last_sent_reply_norm = ""
    last_sent_time = 0.0
    session_click_state = SessionClickState()

    print("Auto reply bot started. Move mouse to top-left corner to abort.")
    while True:
        try:
            now = time.time()
            if now < suppress_until:
                time.sleep(cfg.poll_interval_sec)
                continue
            if not try_click_unread_badge(cfg, session_click_state, now):
                time.sleep(cfg.poll_interval_sec)
                continue
            lines = ocr_lines(ocr_engine, cfg.chat_region)
            if not lines:
                time.sleep(cfg.poll_interval_sec)
                continue

            partner_latest_text = latest_partner_message(lines, cfg, cfg.chat_region)
            if len(partner_latest_text) < cfg.min_text_chars:
                time.sleep(cfg.poll_interval_sec)
                continue
            if (
                last_sent_reply_norm
                and normalize_text(partner_latest_text) == last_sent_reply_norm
                and (now - last_sent_time) < cfg.self_echo_guard_sec
            ):
                if cfg.debug_llm:
                    print("[guard] detected self echo text in partner slot; skip this round.")
                time.sleep(cfg.poll_interval_sec)
                continue

            current_hash = hash_text(partner_latest_text)
            # Never reply to the same partner message twice.
            if current_hash == last_seen_hash:
                time.sleep(cfg.poll_interval_sec)
                continue

            chat_text = build_chat_context(lines)
            if seems_non_chat_capture(chat_text):
                if cfg.debug_llm:
                    print("[ocr] detected non-chat content in capture; skip this round.")
                time.sleep(cfg.poll_interval_sec)
                continue
            prompt = (
                f"对方最新消息：{partner_latest_text}\n\n"
                f"聊天上下文：\n{chat_text}\n\n"
                "现在请根据以上信息生成回复。"
            )
            if not cfg.enable_llm:
                print(f"[llm-disabled] partner_latest={partner_latest_text}")
                if cfg.debug_llm:
                    print(f"[llm-disabled] context:\n{preview_text(chat_text)}")
                last_seen_hash = current_hash
                time.sleep(cfg.poll_interval_sec)
                continue

            reply = call_llm(cfg, prompt)
            if is_echo_reply(reply, partner_latest_text):
                if cfg.debug_llm:
                    print("[llm] echo reply detected, retrying rewrite once.")
                rewrite_prompt = (
                    "你刚才的回复和对方原句重复了。请重新回答，要求：\n"
                    "1) 不要复述对方原句；\n"
                    "2) 用自然中文简短回应（1-2句）；\n"
                    "3) 不确定时回复：收到，稍后回复你。\n\n"
                    f"对方消息：{partner_latest_text}\n"
                    f"上下文：\n{chat_text}"
                )
                reply = call_llm(cfg, rewrite_prompt)
                if is_echo_reply(reply, partner_latest_text):
                    if cfg.debug_llm:
                        print("[llm] echo reply again; skip sending this round.")
                    time.sleep(cfg.poll_interval_sec)
                    continue
            if reply:
                if cfg.real_send:
                    send_message(reply, cfg.input_box_point)
                    print(f"sent: {reply[:60]}")
                else:
                    print(f"[dry-run] would send (real_send=false): {reply}")
                last_seen_hash = current_hash
                last_sent_reply_norm = normalize_text(reply)
                last_sent_time = time.time()
                suppress_until = time.time() + max(0.0, cfg.post_send_cooldown_sec)

            time.sleep(cfg.poll_interval_sec)
        except pyautogui.FailSafeException:
            print("failsafe triggered, exiting.")
            break
        except Exception as exc:
            print(f"error: {exc}")
            time.sleep(cfg.poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen OCR + LLM auto reply bot")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="path to yaml config (default: config.yaml)",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
