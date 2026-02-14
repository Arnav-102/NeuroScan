import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS just in case
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        # Explicitly set MIME types for critical files
        if self.path.endswith('.wasm'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/wasm')
            self.end_headers()
            with open(os.path.join(DIRECTORY, self.path.lstrip('/')), 'rb') as f:
                self.wfile.write(f.read())
            return
        
        if self.path.endswith('.onnx') or self.path.endswith('.data'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()
            with open(os.path.join(DIRECTORY, self.path.lstrip('/')), 'rb') as f:
                self.wfile.write(f.read())
            return
        
        return super().do_GET()

# Registry override for Windows where MIME types might be missing
import mimetypes
mimetypes.init()
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('application/octet-stream', '.onnx') # or model/onnx

print(f"Starting server at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.allow_reuse_address = True
    print("Serving forever...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()
