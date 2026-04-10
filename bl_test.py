#!/usr/bin/env python3
import argparse
import socket
import threading
import time
import sys


RFCOMM_CHANNEL = 4
RECONNECT_DELAY = 3


class PeerNode:
    def _init_(self, my_name: str, peer_mac: str, channel: int):
        self.my_name = my_name
        self.peer_mac = peer_mac
        self.channel = channel

        self.server_sock = None
        self.conn = None
        self.conn_lock = threading.Lock()
        self.stop_event = threading.Event()

    def log(self, msg: str) -> None:
        print(f"[{self.my_name}] {msg}", flush=True)

    def set_connection(self, sock: socket.socket, source: str) -> bool:
        with self.conn_lock:
            if self.conn is not None:
                return False
            self.conn = sock
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
                self.log("Connection closed")

    def get_connection(self):
        with self.conn_lock:
            return self.conn

    def start_server(self) -> None:
        self.server_sock = socket.socket(
            socket.AF_BLUETOOTH,
            socket.SOCK_STREAM,
            socket.BTPROTO_RFCOMM
        )
        # Bind to any local Bluetooth adapter, chosen RFCOMM channel
        self.server_sock.bind((socket.BDADDR_ANY, self.channel))
        self.server_sock.listen(1)
        self.log(f"Listening for RFCOMM on channel {self.channel}")

        while not self.stop_event.is_set():
            try:
                client_sock, client_info = self.server_sock.accept()
                accepted = self.set_connection(client_sock, f"incoming from {client_info[0]}")
                if not accepted:
                    client_sock.close()
            except OSError:
                if not self.stop_event.is_set():
                    self.log("Server socket error")
                break

    def connect_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.get_connection() is not None:
                time.sleep(1)
                continue

            sock = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_STREAM,
                socket.BTPROTO_RFCOMM
            )
            sock.settimeout(8)

            try:
                self.log(f"Trying outbound connect to {self.peer_mac}:{self.channel}")
                sock.connect((self.peer_mac, self.channel))
                sock.settimeout(None)
                accepted = self.set_connection(sock, f"outgoing to {self.peer_mac}")
                if not accepted:
                    sock.close()
            except OSError:
                sock.close()
                time.sleep(RECONNECT_DELAY)

    def recv_loop(self) -> None:
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
                    print(f"\n<< {text}", flush=True)
            except OSError:
                self.clear_connection()
                buffer = b""

    def send_loop(self) -> None:
        self.log("Type messages and press Enter to send. Ctrl+C to quit.")
        while not self.stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if line == "":
                    time.sleep(0.1)
                    continue

                msg = line.rstrip("\n")
                if not msg:
                    continue

                payload = f"{self.my_name}: {msg}\n".encode("utf-8")

                conn = self.get_connection()
                if conn is None:
                    self.log("Not connected yet")
                    continue

                conn.sendall(payload)
            except KeyboardInterrupt:
                self.stop_event.set()
                break
            except OSError:
                self.clear_connection()

    def run(self) -> None:
        threads = [
            threading.Thread(target=self.start_server, daemon=True),
            threading.Thread(target=self.connect_loop, daemon=True),
            threading.Thread(target=self.recv_loop, daemon=True),
        ]

        for t in threads:
            t.start()

        try:
            self.send_loop()
        finally:
            self.stop_event.set()
            self.clear_connection()
            if self.server_sock is not None:
                try:
                    self.server_sock.close()
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Peer-to-peer Bluetooth RFCOMM chat for Arduino UNO Q")
    parser.add_argument("--name", required=True, help="This board's name, e.g. UNO_A")
    parser.add_argument("--peer-mac", required=True, help="Bluetooth MAC address of the other UNO Q")
    parser.add_argument("--channel", type=int, default=RFCOMM_CHANNEL, help="RFCOMM channel to use")
    args = parser.parse_args()

    node = PeerNode(args.name, args.peer_mac.upper(), args.channel)
    node.run()


if __name__ == "__main__":
    main()