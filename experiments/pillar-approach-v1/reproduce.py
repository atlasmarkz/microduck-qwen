import json
import math
from pathlib import Path

from microduck_qwen.agent import AgentPlanner, run_instruction
from microduck_qwen.skills import SkillExecutor, memory_state
from microduck_qwen.sim_data import request
from microduck_qwen.sim_runner import start_scene, stop_process
from microduck_qwen.vision import capture, propose_targets, save_frame

output = Path('runs/pillar-approach-reproduce')
output.mkdir(parents=True, exist_ok=False)
process = log = executor = None
report = {}
try:
    process, log, info, failures = start_scene(output, 92001, 'none', 8062, collect_gt=True, yellow_course=True)
    executor = SkillExecutor(info, output / 'setup')
    initial = executor.healthy()
    scene = executor.scene
    nearest = min(scene['obstacles'], key=lambda item: math.dist(initial['trunk'][:2], [item['x'], item['y']]))
    frame = capture(info['port'])
    candidates = propose_targets(frame)
    image = save_frame(output, frame, candidates)
    report['initial_candidates'] = candidates
    report['nearest_ground_truth_for_scoring_only'] = nearest
    planner = AgentPlanner(Path('adapters/agent-yellow-v1'))
    report['ambiguous'] = planner.predict('走到黄色的柱子旁边停下', image, memory_state(info, initial, []))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    executor.close()
    executor = None
    answers = iter(['走到离我最近的黄色柱子旁边停下'])
    report['episode'] = run_instruction(planner, '走到黄色的柱子旁边停下', info, ask=lambda question: next(answers))
    observer = SkillExecutor(info, output / 'score')
    try:
        final = observer.healthy()
        report['final_distance_to_nearest_pillar_m'] = math.dist(final['trunk'][:2], [nearest['x'], nearest['y']])
    finally:
        observer.close()
    report['contacts'] = request(info['port'], 'collection_stats')
    assert report['episode']['status'] in ('model_stop', 'model_done')
    assert len(report['episode']['interactions']) == 1
    assert report['episode']['steps'][0]['execution']['no_motion']
    assert 0.30 <= report['final_distance_to_nearest_pillar_m'] <= 0.60
    assert report['contacts']['steps_with_scene_contact'] == 0
finally:
    if executor:
        executor.close()
    if process:
        stop_process(process)
    if log:
        log.close()
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
