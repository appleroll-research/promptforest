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
    
    # Config Command
    config_parser = subparsers.add_parser("config", help="View effective configuration")
    config_parser.add_argument("config_path", type=str, nargs='?', help="Path to configuration file to load and view")

    args = parser.parse_args()
    
    # helper to load config
    def get_user_config(path):
        # If user explicitly provided --config, use it
        if path:
            print(f"Loading configuration from {path}...")
            return load_config(path)
        # If not, check if config.yaml exists in current dir
        elif os.path.exists("config.yaml"):
             print("Loading config from current directory (config.yaml)...")
             return load_config("config.yaml")
        # Otherwise use defaults
        return load_config(None)

    if args.command == "serve":
        from .server import run_server
        cfg = get_user_config(args.config)
        run_server(port=args.port, config=cfg)

    elif args.command == "check":
        from .lib import EnsembleGuard
        cfg = get_user_config(args.config)
        try:
            print(f"Loading PromptForest...")
            guard = EnsembleGuard(config=cfg)
            print(f"Device: {guard.device_used}")
            # Suppress loading messages if config is doing stats only? 
            # The user asked for stats in output. lib.py handles that in check_prompt result.
            result = guard.check_prompt(args.prompt)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif args.command == "config":
        # Load and display configuration
        cfg = get_user_config(args.config_path)
        print(json.dumps(cfg, indent=2))

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
