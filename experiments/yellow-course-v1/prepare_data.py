import json
from pathlib import Path
import shutil

from microduck_qwen.agent_data import load_splits, write_splits
from microduck_qwen.sim_data import digest

output = Path('data/agent-yellow-mixed-v1')
output.mkdir(exist_ok=False)
splits = {key: [] for key in ('train', 'valid', 'test')}
parents = {}
yellow_train = []
for label, source in [('old', Path('data/agent-hard-v3')), ('yellow', Path('data/yellow-course-v1'))]:
    _, data = load_splits(source)
    shutil.copytree(source, output / label)
    parents[label] = digest(source / 'manifest.json')
    for split, rows in data.items():
        prefixed = [dict(row, image=label + '/' + row['image']) for row in rows]
        splits[split].extend(prefixed)
        if label == 'yellow' and split == 'train':
            yellow_train = prefixed
manifest = {'format': 'simulation_agent_v1', 'status': 'completed', 'splits': {},
            'parent_manifests': parents,
            'method': 'Disjoint old tasks plus physically verified yellow bypass demonstrations; no reward-selected labels'}
write_splits(output, manifest, splits)
(output / 'yellow-replay.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in yellow_train), encoding='utf-8')
load_splits(output)
print({key: len(rows) for key, rows in splits.items()})
