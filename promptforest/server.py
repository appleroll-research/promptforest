"""
Server module for PromptForest. Sets up an HTTP server to handle inference requests
"""

import http.server
import socketserver
import json
import sys
import time
from .lib import PFEnsemble

PORT = 8000
ensemble = None

class PFRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for inference."""
        if self.path == '/analyze':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self._send_json(400, {'error': 'Empty body'})
                    return
                    
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                prompt = data.get('prompt')
                
                if not prompt:
                    self._send_json(400, {'error': 'Field "prompt" is required'})
                    return

                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Received prompt \"{prompt}\"")
                
                result = ensemble.check_prompt(prompt)
                self._send_json(200, result)
                
            except json.JSONDecodeError:
                self._send_json(400, {'error': 'Invalid JSON'})
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Server Error: {e}")
                self._send_json(500, {'error': str(e)})
        else:
            self._send_json(404, {'error': 'Endpoint not found.'})

    def do_GET(self):
        if self.path == '/health':
            device = ensemble.device_used if ensemble else "unknown"
            self._send_json(200, {'status': 'ok', 'device': device})
        else:
            self._send_json(404, {'error': 'Not Found'})

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True

def run_server(port=8000, config=None):
    global ensemble
    print(f"Initializing PromptForest...")
    try:
        ensemble = PFEnsemble(config=config)
        print(f"Device: {ensemble.device_used}")
        print("Warming up...")
        ensemble.check_prompt("warmup")
        print("Ready.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    print(f"\nListening on http://localhost:{port}")
    socketserver.TCPServer.allow_reuse_address = True
    
    with ThreadedHTTPServer(("", port), PFRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down. Goodbye...")
            httpd.shutdown()
            httpd.server_close()
