import time

import pyautogui


def main() -> None:
    pyautogui.FAILSAFE = True
    print("Move mouse to inspect coordinates. Ctrl+C to stop.")
    while True:
        x, y = pyautogui.position()
        print(f"\rX={x:4d} Y={y:4d}", end="", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
