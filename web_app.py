import json
import os
import random
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Bluetooth peer configuration — set via environment variables before launch:
#   PEERS_FILE   path to JSON file listing peers (default: peers.json)
#   BT_CHANNEL   RFCOMM channel                 (default: 4)
#   BT_INTERVAL  seconds between messages       (default: 5)
#   BT_MESSAGE   fixed message to send          (default: PING)
#
# peers.json format:
#   { "peers": [ {"name": "NODE_B", "mac": "AA:BB:CC:DD:EE:FF"}, ... ] }
# ---------------------------------------------------------------------------
_PEER_NAME = socket.gethostname()
_PEERS_FILE = os.getenv("PEERS_FILE", "peers.json")
_BT_CHANNEL = int(os.getenv("BT_CHANNEL", "4"))
_BT_INTERVAL = int(os.getenv("BT_INTERVAL", "5"))
_BT_MESSAGE = os.getenv("BT_MESSAGE", "PING")


def _load_peer_macs(path: str) -> list[str]:
    """Load peer MAC addresses from a JSON config file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return [str(p["mac"]).upper() for p in data.get("peers", []) if p.get("mac")]
    except FileNotFoundError:
        print(f"[config] peers file '{path}' not found — no outbound peers", flush=True)
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[config] failed to parse '{path}': {e}", flush=True)
        return []

RECONNECT_DELAY = 3


# ---------------------------------------------------------------------------
# Peer-to-peer Bluetooth node (runs in background threads)
# ---------------------------------------------------------------------------

class PeerNode:
    def __init__(self, my_name: str, peer_macs: list[str], channel: int,
                 interval: int, fixed_message: str):
        self.my_name = my_name
        self.peer_macs = peer_macs
        self.channel = channel
        self.interval = interval
        self.fixed_message = fixed_message

        self.server_sock = None
        self.conn = None
        self.active_peer_mac: str | None = None
        self.conn_lock = threading.Lock()
        self.stop_event = threading.Event()

        self._counter_lock = threading.Lock()
        self.messages_sent = 0
        self.messages_received = 0

    # -- helpers ------------------------------------------------------------

    def log(self, msg: str) -> None:
        print(f"[{self.my_name}] {msg}", flush=True)

    @property
    def is_connected(self) -> bool:
        with self.conn_lock:
            return self.conn is not None

    @property
    def messages_exchanged(self) -> int:
        with self._counter_lock:
            return self.messages_sent + self.messages_received

    def set_connection(self, sock: socket.socket, source: str, peer_mac: str | None = None) -> bool:
        with self.conn_lock:
            if self.conn is not None:
                return False
            self.conn = sock
            self.active_peer_mac = peer_mac
            self.log(f"Connected via {source}")
            return True

    def clear_connection(self) -> None:
        with self.conn_lock:
            if self.conn is not None:
                try:
                    self.conn.close()
                except OSError:
                    pass
                self.conn = None
                self.active_peer_mac = None
                self.log("Connection closed")

    def get_connection(self):
        with self.conn_lock:
            return self.conn

    # -- threads ------------------------------------------------------------

    def _start_server(self) -> None:
        try:
            self.server_sock = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_STREAM,
                socket.BTPROTO_RFCOMM,
            )
            self.server_sock.bind((socket.BDADDR_ANY, self.channel))
            self.server_sock.listen(1)
            self.log(f"Listening for RFCOMM on channel {self.channel}")
        except OSError as e:
            self.log(f"Could not start Bluetooth server: {e}")
            return

        while not self.stop_event.is_set():
            try:
                client_sock, client_info = self.server_sock.accept()
                src_mac = client_info[0].upper()
                accepted = self.set_connection(
                    client_sock, f"incoming from {src_mac}", src_mac
                )
                if not accepted:
                    client_sock.close()
            except OSError:
                if not self.stop_event.is_set():
                    self.log("Server socket error")
                break

    def _connect_loop(self, target_mac: str) -> None:
        while not self.stop_event.is_set():
            if self.get_connection() is not None:
                time.sleep(1)
                continue

            sock = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_STREAM,
                socket.BTPROTO_RFCOMM,
            )
            sock.settimeout(8)

            try:
                self.log(f"Trying outbound connect to {target_mac}:{self.channel}")
                sock.connect((target_mac, self.channel))
                sock.settimeout(None)
                accepted = self.set_connection(sock, f"outgoing to {target_mac}", target_mac)
                if not accepted:
                    sock.close()
            except OSError:
                sock.close()
                time.sleep(RECONNECT_DELAY)

    def _recv_loop(self) -> None:
        buffer = b""

        while not self.stop_event.is_set():
            conn = self.get_connection()
            if conn is None:
                time.sleep(0.2)
                continue

            try:
                data = conn.recv(1024)
                if not data:
                    self.clear_connection()
                    buffer = b""
                    continue

                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    self.log(f"<< {text}")
                    with self._counter_lock:
                        self.messages_received += 1
            except OSError:
                self.clear_connection()
                buffer = b""

    def _periodic_send_loop(self) -> None:
        self.log(
            f"Periodic sender ready — will send '{self.fixed_message}' "
            f"every {self.interval}s once connected"
        )
        while not self.stop_event.is_set():
            conn = self.get_connection()
            if conn is None:
                time.sleep(0.2)
                continue

            payload = f"{self.my_name}: {self.fixed_message}\n".encode("utf-8")
            try:
                conn.sendall(payload)
                with self._counter_lock:
                    self.messages_sent += 1
                self.log(f">> {self.fixed_message}")
            except OSError:
                self.clear_connection()
                continue

            # Interruptible sleep
            for _ in range(self.interval * 10):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._start_server, daemon=True).start()
        for mac in self.peer_macs:
            threading.Thread(target=self._connect_loop, args=(mac,), daemon=True).start()
        for target in (self._recv_loop, self._periodic_send_loop):
            threading.Thread(target=target, daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        self.clear_connection()
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

peer_node = PeerNode(
    my_name=_PEER_NAME,
    peer_macs=_load_peer_macs(_PEERS_FILE),
    channel=_BT_CHANNEL,
    interval=_BT_INTERVAL,
    fixed_message=_BT_MESSAGE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    peer_node.start()
    yield
    peer_node.stop()


app = FastAPI(title="Image Classifier", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def mock_classify(image_bytes: bytes) -> int:
    """Mock AI model — replace with real inference later."""
    return random.randint(0, 9)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext != ".ppm":
        return {"error": f"Unsupported file type '{ext}'. Only .ppm files are accepted."}

    image_bytes = await file.read()
    predicted_class = mock_classify(image_bytes)

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
    }


@app.get("/peer-status")
async def peer_status():
    return {
        "connected": peer_node.is_connected,
        "active_peer_mac": peer_node.active_peer_mac,
        "candidate_peer_macs": peer_node.peer_macs,
        "messages_sent": peer_node.messages_sent,
        "messages_received": peer_node.messages_received,
        "messages_exchanged": peer_node.messages_exchanged,
    }
