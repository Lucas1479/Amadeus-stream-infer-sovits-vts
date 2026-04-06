"""render/server.py — 本地静态资产 HTTP 服务器

用途：QWebEngineView 加载 http://127.0.0.1:{port}/render/web/index.html，
同时使所有项目文件（图片、模型等）可从浏览器上下文访问。
"""
import http.server
import threading
import socket
from pathlib import Path


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """添加 CORS 头，允许 PixiJS / pixi-live2d-display 跨源加载资产。"""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):  # 静默日志
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


class AssetServer:
    """以 root 为 document root 启动本地 HTTP 服务器。

    Parameters
    ----------
    root:       服务器根目录（项目根目录）
    start_port: 首选端口，若被占用则自动递增至 start_port+20
    """

    def __init__(self, root: Path, start_port: int = 17777):
        self.root = Path(root)
        self.start_port = start_port
        self.port: int = -1
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------

    def start(self) -> int:
        """启动服务器并返回实际监听端口。"""
        handler = _make_handler(self.root)
        for p in range(self.start_port, self.start_port + 20):
            if _port_free(p):
                self._server = http.server.HTTPServer(("127.0.0.1", p), handler)
                self.port = p
                break
        else:
            raise OSError(f"No free port in [{self.start_port}, {self.start_port+20})")

        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="AssetServer"
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(root: Path):
    """工厂：创建固定 directory 的 handler 类（避免 os.chdir）。"""
    root_str = str(root)

    class _Handler(_CORSHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root_str, **kwargs)

    return _Handler


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
