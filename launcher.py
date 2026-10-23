"""Control panel that starts Ollama, server.py and ngrok and shows their status.

Run:  .venv\\Scripts\\pythonw.exe launcher.py   (or use the desktop shortcut)

Environment variables (all optional):
    PORT          port server.py listens on (default 8000)
    NGROK_DOMAIN  reserved ngrok domain, e.g. "my-app.ngrok-free.app"
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import scrolledtext, ttk

HERE = Path(__file__).resolve().parent
PORT = os.environ.get("PORT", "8000")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"

COLORS = {"running": "#16a34a", "starting": "#d97706", "stopped": "#9ca3af", "error": "#dc2626"}


def get_json(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


class Service:
    """One background program plus a way to ask whether it's healthy."""

    def __init__(self, name, cmd, health_url):
        self.name, self.cmd, self.health_url = name, cmd, health_url
        self.proc = None
        self.detail = ""

    def start(self, log):
        if self.is_healthy():
            log(f"[{self.name}] already running, not starting another copy")
            return
        exe = shutil.which(self.cmd[0]) or self.cmd[0]
        try:
            self.proc = subprocess.Popen(
                [exe, *self.cmd[1:]], cwd=HERE, creationflags=NO_WINDOW,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except OSError as e:
            log(f"[{self.name}] failed to start: {e}")
            return
        log(f"[{self.name}] started (pid {self.proc.pid})")
        threading.Thread(target=self._pump, args=(log,), daemon=True).start()

    def _pump(self, log):
        for line in self.proc.stdout:
            log(f"[{self.name}] {line.rstrip()}")

    def stop(self, log):
        if self.proc and self.proc.poll() is None:
            # /T also kills children (Ollama spawns model runners).
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=NO_WINDOW)
            log(f"[{self.name}] stopped")
        self.proc = None

    def is_healthy(self):
        return get_json(self.health_url) is not None

    def status(self):
        if self.is_healthy():
            return "running"
        if self.proc is None:
            return "stopped"
        if self.proc.poll() is not None:
            self.detail = f"exited with code {self.proc.returncode}"
            return "error"
        return "starting"


class Ngrok(Service):
    def is_healthy(self):
        data = get_json(self.health_url)
        tunnels = (data or {}).get("tunnels") or []
        url = next((t["public_url"] for t in tunnels if t["public_url"].startswith("https")), None)
        self.detail = url or ""
        return url is not None


def build_services():
    ngrok_cmd = ["ngrok", "http", PORT, "--log", "stdout"]
    if os.environ.get("NGROK_DOMAIN"):
        ngrok_cmd += ["--domain", os.environ["NGROK_DOMAIN"]]
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    return [
        Service("Ollama", ["ollama", "serve"], "http://127.0.0.1:11434/api/tags"),
        Service("Server", [py, "server.py"], f"http://127.0.0.1:{PORT}/health"),
        Ngrok("ngrok", ngrok_cmd, "http://127.0.0.1:4040/api/tunnels"),
    ]


class App:
    def __init__(self, root):
        self.root = root
        self.services = build_services()
        self.logs = queue.Queue()
        self.rows = {}

        root.title("Pronunciation Server")
        root.geometry("720x480")
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)

        for i, svc in enumerate(self.services):
            dot = tk.Canvas(frame, width=14, height=14, highlightthickness=0)
            dot.grid(row=i, column=0, padx=(0, 8))
            oval = dot.create_oval(2, 2, 12, 12, fill=COLORS["stopped"], outline="")
            ttk.Label(frame, text=svc.name, width=8, font=("Segoe UI", 10, "bold")).grid(row=i, column=1, sticky="w")
            status = ttk.Label(frame, text="stopped", width=40)
            status.grid(row=i, column=2, sticky="w")
            btn = ttk.Button(frame, text="Start", width=8, command=lambda s=svc: self.toggle(s))
            btn.grid(row=i, column=3, pady=2)
            self.rows[svc.name] = (dot, oval, status, btn)

        n = len(self.services)
        url_row = ttk.Frame(frame)
        url_row.grid(row=n, column=0, columnspan=4, sticky="we", pady=(10, 4))
        ttk.Label(url_row, text="Public URL:").pack(side="left")
        self.url_var = tk.StringVar(value="-")
        ttk.Entry(url_row, textvariable=self.url_var, state="readonly", width=50).pack(side="left", padx=6)
        ttk.Button(url_row, text="Copy", command=self.copy_url).pack(side="left")

        bar = ttk.Frame(frame)
        bar.grid(row=n + 1, column=0, columnspan=4, sticky="w", pady=4)
        ttk.Button(bar, text="Start all", command=self.start_all).pack(side="left")
        ttk.Button(bar, text="Stop all", command=self.stop_all).pack(side="left", padx=6)

        self.log_box = scrolledtext.ScrolledText(frame, height=14, font=("Consolas", 9), state="disabled")
        self.log_box.grid(row=n + 2, column=0, columnspan=4, sticky="nsew", pady=(6, 0))
        frame.columnconfigure(2, weight=1)
        frame.rowconfigure(n + 2, weight=1)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.drain_logs)
        self.poll()
        self.start_all()

    def log(self, msg):
        self.logs.put(msg)  # called from worker threads; Tk is only touched in drain_logs

    def drain_logs(self):
        lines = []
        while not self.logs.empty():
            lines.append(self.logs.get_nowait())
        if lines:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", "\n".join(lines) + "\n")
            self.log_box.delete("1.0", "end-2000l")  # keep the last 2000 lines
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(200, self.drain_logs)

    def in_background(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def toggle(self, svc):
        if svc.proc and svc.proc.poll() is None:
            self.in_background(svc.stop, self.log)
        else:
            self.in_background(svc.start, self.log)

    def start_all(self):
        def run():
            for svc in self.services:  # in order: Ollama, then server, then ngrok
                svc.start(self.log)
        self.in_background(run)

    def stop_all(self):
        def run():
            for svc in reversed(self.services):
                svc.stop(self.log)
        self.in_background(run)

    def copy_url(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.url_var.get())

    def poll(self):
        def check():
            results = [(svc, svc.status()) for svc in self.services]
            self.root.after(0, self.show, results)
        self.in_background(check)
        self.root.after(2000, self.poll)

    def show(self, results):
        for svc, state in results:
            dot, oval, status, btn = self.rows[svc.name]
            dot.itemconfigure(oval, fill=COLORS[state])
            text = state
            if state == "running" and svc.proc is None:
                text += " (started outside this app)"
            elif state == "starting" and svc.name == "Server":
                text += " (loading model...)"
            elif state == "error":
                text += f" - {svc.detail}"
            status.configure(text=text)
            ours = svc.proc is not None and svc.proc.poll() is None
            btn.configure(text="Stop" if ours else "Start", state="normal" if ours or state != "running" else "disabled")
            if isinstance(svc, Ngrok):
                self.url_var.set(svc.detail if state == "running" else "-")

    def on_close(self):
        for svc in reversed(self.services):
            svc.stop(self.log)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
