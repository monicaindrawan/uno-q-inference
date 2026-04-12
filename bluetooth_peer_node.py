import socket
import threading
import time
import json

RECONNECT_DELAY = 3

def load_peer_macs(path: str) -> list[str]:
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

class PeerNode:
    def __init__(self, my_name: str, peer_macs: list[str], channel: int,
                 on_message=None):
        self.my_name = my_name
        self.peer_macs = peer_macs
        self.channel = channel
        self.on_message = on_message  # callable(text: str) -> None

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
                    with self._counter_lock:
                        self.messages_received += 1
                    if self.on_message is not None:
                        self.on_message(text)
            except OSError:
                self.clear_connection()
                buffer = b""

    def send(self, message: str) -> bool:
        """Send a message immediately over the current connection.

        Returns True if the message was sent, False if not connected.
        """
        conn = self.get_connection()
        if conn is None:
            return False
        payload = f"{message}\n".encode("utf-8")
        try:
            conn.sendall(payload)
            with self._counter_lock:
                self.messages_sent += 1
            return True
        except OSError:
            self.clear_connection()
            return False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._start_server, daemon=True).start()
        for mac in self.peer_macs:
            threading.Thread(target=self._connect_loop, args=(mac,), daemon=True).start()
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        self.clear_connection()
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except OSError:
                pass