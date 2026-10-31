"""Publish the saved pilot reports with workspace paths made relative."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'runs/local-review-20261031'
output = Path(__file__).parent

def save(name, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    text = text.replace(str(root).replace('\\', '\\\\'), '.').replace(root.as_posix(), '.')
    (output / name).write_text(text + '\n', encoding='utf-8')

report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
save('report.json', report)
for index in range(4):
    data = json.loads((source / f'task-{index:02d}/episode.json').read_text(encoding='utf-8'))
    save(f'task-{index:02d}.json', data)
