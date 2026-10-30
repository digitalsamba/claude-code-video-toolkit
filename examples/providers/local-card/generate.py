"""An offline provider adapter example; no third-party packages or services."""
import argparse
import html
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.input.read_text(encoding='utf-8'))
if args.output.suffix.lower() != '.svg':
    parser.error('local-card requires a .svg output')
prompt = html.escape(request['prompt'][:70])
args.output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080"><rect width="100%" height="100%" fill="#13251f"/><text x="100" y="540" fill="#fff8e9" font-family="serif" font-size="52">{prompt}</text></svg>', encoding='utf-8')
print(json.dumps({'version': 1, 'output': str(args.output)}))
