#!/usr/bin/env python3
"""Idempotently register the production commands, tools and template."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / '_internal/toolkit-registry.json'
    data = json.loads(target.read_text(encoding='utf-8'))
    entries = {
        'project': ('tools/project.py', 'Validate and migrate versioned project manifests', 'uv run tools/project.py validate PROJECT --json'),
        'build': ('tools/build.py', 'Preview or run checkpointed content-aware video builds with cost limits', 'uv run tools/build.py --project PROJECT --json'),
        'budget': ('tools/budget.py', 'Inspect estimated spending and reconcile provider charges', 'uv run tools/budget.py --project PROJECT'),
        'review': ('tools/review.py', 'Local storyboard, narration previews, QC findings, notes and approvals', 'uv run tools/review.py PROJECT'),
        'asset_history': ('tools/asset_history.py', 'Record, search and restore immutable asset versions', 'uv run tools/asset_history.py --project PROJECT list'),
        'cues': ('tools/cues.py', 'Resolve named narration phrases to motion timestamps', 'uv run tools/cues.py --help'),
        'variants': ('tools/variants.py', 'Validate and materialize language and campaign video variants', 'uv run tools/variants.py --project PROJECT --spec variants.json'),
        'subtitles': ('tools/subtitles.py', 'Export word captions to downloadable SRT/WebVTT', 'uv run tools/subtitles.py WORDS.json --output out/captions.srt'),
        'provider': ('tools/provider.py', 'Preview or run a trusted local JSON provider adapter', 'uv run tools/provider.py --help'),
        'extensions': ('tools/extensions.py', 'Discover, validate and scaffold local template/provider contracts', 'uv run tools/extensions.py list'),
    }
    for name, (path, description, usage) in entries.items():
        if not (ROOT / path).is_file():
            raise SystemExit(f'Missing registered tool: {path}')
        data['tools'][name] = {'path': path, 'description': description, 'usage': usage, 'status': 'beta', 'created': '2026-10-31'}
    for name, description in {
        'build-video': 'Preview, execute or resume a production build with selective regeneration and cost limits',
        'review-video': 'Review a local storyboard with media previews, QC findings and version-bound approvals',
    }.items():
        data['commands'][name] = {'path': f'.claude/commands/{name}.md', 'description': description, 'status': 'beta', 'created': '2026-10-31'}
    data['templates']['storyboard'] = {'path': 'templates/storyboard/', 'description': 'Manifest-driven responsive scenes for landscape, portrait and square; supports language/campaign props', 'status': 'beta', 'engine': 'remotion', 'formats': ['16:9', '9:16', '1:1'], 'created': '2026-10-31'}
    data['examples']['narration-cues'] = {'path': 'examples/narration-cues/', 'description': 'Speech-linked Stage cursor/camera motion using generated word cues', 'created': '2026-10-31'}
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
