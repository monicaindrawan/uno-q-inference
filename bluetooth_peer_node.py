"""
Bluetooth Peer Node — RFCOMM Socket Manager
============================================

Manages a single Bluetooth RFCOMM connection between two camera nodes so they
can exchange inference requests and embeddings during collaborative inference.

Architecture:
    - Each node runs both a server thread (waits for inbound connections) AND
      one outbound connect-loop thread per configured peer MAC address.
    - Only ONE connection is held at a time (first to succeed wins); duplicates
      are rejected via a threading lock.
    - A single recv-loop thread continuously reads newline-delimited JSON
      messages from the active socket and dispatches them to on_message().

Threading model:
    _start_server()      — accepts inbound RFCOMM connections
    _connect_loop(mac)   — retries outbound connections to a peer MAC
    _recv_loop()         — reads messages from the active socket
    send()               — can be called from any thread safely

Connection state is protected by conn_lock (read/write) and
_counter_lock (message counters only).
"""

import socket
import threading
import time
import json

RECONNECT_DELAY = 3   # seconds to wait before retrying a failed outbound connect

# =============================================================================
# Config Loader
# =============================================================================

def load_peer_macs(path: str) -> list[str]:
    """Load peer MAC addresses from a JSON config file.

    Expected format:
        { "peers": [ { "mac": "AA:BB:CC:DD:EE:FF" }, ... ] }

    Returns an empty list on missing file or parse error so the node can still
    start in solo mode without crashing.
    """
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


# =============================================================================
# PeerNode
# Manages the lifecycle of a single RFCOMM connection to one peer node.
# =============================================================================

class PeerNode:
    def __init__(self, my_name: str, peer_macs: list[str], channel: int,
                 on_message=None):
        """
        Args:
            my_name    : human-readable label used in log messages
            peer_macs  : list of Bluetooth MAC addresses to try connecting to
            channel    : RFCOMM channel (must match on both ends)
            on_message : callback(text: str) invoked on each received message
        """
        self.my_name = my_name
        self.peer_macs = peer_macs
        self.channel = channel
        self.on_message = on_message  # callable(text: str) -> None

        # Active socket and its metadata; protected by conn_lock
        self.server_sock = None
        self.conn = None
        self.active_peer_mac: str | None = None
        self.conn_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Message counters; protected by _counter_lock
        self._counter_lock = threading.Lock()
        self.messages_sent = 0
        self.messages_received = 0

    # =========================================================================
    # Helpers
    # =========================================================================

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
        """Atomically claim the connection slot.

        Returns False (and leaves sock unclosed) if a connection is already
        held — the caller must close the socket itself.
        """
        with self.conn_lock:
            if self.conn is not None:
                return False
            self.conn = sock
            self.active_peer_mac = peer_mac
            self.log(f"Connected via {source}")
            return True

    def clear_connection(self) -> None:
        """Close and release the active socket, resetting connection state."""
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
        """Return the active socket (or None) under the connection lock."""
        with self.conn_lock:
            return self.conn

    # =========================================================================
    # Background Threads
    # =========================================================================

    def _start_server(self) -> None:
        """Listen for inbound RFCOMM connections on self.channel.

        Runs forever until stop_event is set. Accepts the first connection
        that arrives; any further attempts while a connection is held are
        immediately closed (we only support one peer at a time).
        """
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
        """Continuously attempt outbound RFCOMM connections to target_mac.

        Backs off by RECONNECT_DELAY seconds on each failed attempt.
        Skips immediately if a connection is already active (set by either
        this loop or the server thread).
        """
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
        """Read newline-delimited messages from the active socket.

        Uses a byte buffer to handle partial TCP/RFCOMM frames correctly.
        Each complete line (decoded as UTF-8) is passed to on_message().
        Clears the connection on EOF or socket error.
        """
        buffer = b""

        while not self.stop_event.is_set():
            conn = self.get_connection()
            if conn is None:
                time.sleep(0.2)
                continue

            try:
                data = conn.recv(1024)
                if not data:
                    # Peer closed the connection
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

    # =========================================================================
    # Public API
    # =========================================================================

    def send(self, message: str) -> bool:
        """Send a newline-terminated message over the current connection.

        Thread-safe; can be called from any thread.
        Returns True if sent, False if not connected or the socket errored.
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

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Spin up server, connect-loop(s), and recv-loop as daemon threads."""
        threading.Thread(target=self._start_server, daemon=True).start()
        for mac in self.peer_macs:
            threading.Thread(target=self._connect_loop, args=(mac,), daemon=True).start()
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def stop(self) -> None:
        """Signal all threads to exit and close open sockets."""
        self.stop_event.set()
        self.clear_connection()
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except OSError:
                pass
