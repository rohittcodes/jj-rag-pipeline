#!/usr/bin/env python
"""
JustJosh RAG Pipeline CLI
Unified command-line interface for all operations.

Usage:
    python cli.py setup --all
    python cli.py ingest --all
    python cli.py sync --all
    python cli.py api
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_script(script_name, args):
    """Run a script with uv."""
    script_path = project_root / 'scripts' / f'{script_name}.py'
    cmd = ['uv', 'run', 'python', str(script_path)] + args
    return subprocess.run(cmd).returncode


def run_api(args):
    """Start the API server."""
    cmd = ['uv', 'run', 'uvicorn', 'src.api.main:app']
    
    # Add default flags if not provided
    if '--reload' not in args and '--no-reload' not in args:
        cmd.append('--reload')
    if '--host' not in args:
        cmd.extend(['--host', '0.0.0.0'])
    if '--port' not in args:
        cmd.extend(['--port', '8000'])
    
    cmd.extend(args)
    return subprocess.run(cmd).returncode


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands:")
        print("  setup     - Database setup operations")
        print("  ingest    - Content ingestion & embeddings")
        print("  sync      - External data syncing")
        print("  api       - Start API server")
        print("\nUse 'python cli.py <command> --help' for more info")
        return 1
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == 'api':
        return run_api(args)
    elif command in ['setup', 'ingest', 'sync']:
        return run_script(command, args)
    else:
        print(f"Unknown command: {command}")
        print("Available: setup, ingest, sync, api")
        return 1


if __name__ == '__main__':
    sys.exit(main())
