"""
NeuroForge-3D Local Development Server
Lightweight HTTP server with proper MIME types for ES modules and Web Workers.
"""
import http.server
import socketserver
import os
import sys

PORT = 8080

class NeuroForgeHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and cross-origin isolation headers for high-performance SharedArrayBuffer if needed
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def guess_type(self, path):
        if path.endswith('.js') or path.endswith('.mjs'):
            return 'application/javascript'
        if path.endswith('.css'):
            return 'text/css'
        if path.endswith('.html'):
            return 'text/html'
        if path.endswith('.json'):
            return 'application/json'
        return super().guess_type(path)

def run():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    
    port = PORT
    for attempt in range(5):
        try:
            with socketserver.TCPServer(("", port), NeuroForgeHTTPHandler) as httpd:
                print(f"[NeuroForge-3D Server] Running at http://localhost:{port}/")
                sys.stdout.flush()
                httpd.serve_forever()
        except OSError as e:
            if "address already in use" in str(e).lower() or e.errno == 10048 or e.errno == 98:
                port += 1
            else:
                raise e

if __name__ == '__main__':
    run()
