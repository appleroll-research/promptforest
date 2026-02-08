import argparse
import sys
import json
import os
from .config import load_config

def main():

    parser = argparse.ArgumentParser(description="PromptForest: Ensemble Prompt Injection Detection")
    subparsers = parser.add_subparsers(dest="command")

    # Serve Command
    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--config", type=str, default=None, help="Path to configuration file")

    # Check Command (One-off inference)
    check_parser = subparsers.add_parser("check", help="Check a single prompt")
    check_parser.add_argument("prompt", type=str, help="The prompt to analyze")
    check_parser.add_argument("--config", type=str, default=None, help="Path to configuration file")

    args = parser.parse_args()
    
    # helper to load config
    def get_user_config(path):
        if path:
            print(f"Loading configuration from {path}...")
            return load_config(path)
        return load_config(None)

    if args.command == "serve":
        from .server import run_server
        cfg = get_user_config(args.config)
        run_server(port=args.port, config=cfg)

    elif args.command == "check":
        from .lib import PFEnsemble
        cfg = get_user_config(args.config)
        try:
            print(f"Loading PromptForest...")
            guard = PFEnsemble(config=cfg)
            print(f"Device: {guard.device_used}")
            result = guard.check_prompt(args.prompt)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()
