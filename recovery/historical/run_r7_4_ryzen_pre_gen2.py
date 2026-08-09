from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cpu_name() -> str:
    if os.name == 'nt':
        try:
            out = subprocess.check_output(
                ['powershell', '-NoProfile', '-Command',
                 '(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name) -join " | "'],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or platform.machine()


def total_ram_bytes() -> int | None:
    if os.name == 'nt':
        try:
            out = subprocess.check_output(
                ['powershell', '-NoProfile', '-Command',
                 '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(out)
        except Exception:
            return None
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def git_info(repo: Path) -> dict[str, Any]:
    result: dict[str, Any] = {'available': False}
    try:
        top = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '--show-toplevel'], text=True, stderr=subprocess.DEVNULL).strip()
        head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
        result.update({'available': True, 'top': top, 'head': head, 'dirty': dirty})
    except Exception:
        pass
    return result


def torch_info() -> dict[str, Any]:
    try:
        import torch
        return {
            'available': True,
            'version': torch.__version__,
            'num_threads': torch.get_num_threads(),
            'num_interop_threads': torch.get_num_interop_threads(),
            'cuda_available': bool(torch.cuda.is_available()),
        }
    except Exception as exc:
        return {'available': False, 'error': repr(exc)}


def find_one(repo: Path, candidates: list[str]) -> str | None:
    for rel in candidates:
        if (repo / rel).exists():
            return rel.replace('\\', '/')
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description='SpinCore R7.4 Ryzen9 gate runner')
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = (args.out or (repo / 'validation' / 'R7_4_RYZEN_REPORT.json')).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cpu = cpu_name()
    ram = total_ram_bytes()

    required_groups = {
        'cpp_authoritative_game': [
            'include/spincore/spin_traversal_state.hpp',
            'src/spin_traversal_state.cpp',
            'include/spincore/betting_engine.hpp',
            'src/betting_engine.cpp',
        ],
        'neural_stack': [
            'python/spincore/neural.py',
            'spincore/neural.py',
            'python/spincore/neural_models.py',
        ],
        'deep_cfr_loop': [
            'python/spincore/deep_cfr.py',
            'spincore/deep_cfr.py',
            'tools/run_deep_cfr.py',
        ],
        'r7_metrics_or_pilot': [
            'tools/r7_pilot.py',
            'tools/run_r7_pilot.py',
            'python/spincore/r7.py',
            'spincore/r7.py',
        ],
    }

    found: dict[str, str | None] = {}
    missing: list[str] = []
    for group, candidates in required_groups.items():
        hit = find_one(repo, candidates)
        found[group] = hit
        if hit is None:
            missing.append(group)

    version_files = {}
    for rel in ['VERSION.json', 'STATUS.json', 'ROADMAP.md']:
        p = repo / rel
        if p.exists():
            version_files[rel] = {'sha256': sha256_file(p), 'size': p.stat().st_size}

    report: dict[str, Any] = {
        'schema': 'SPINCORE_R7_4_RYZEN_REPORT_V1',
        'generated_at_unix': time.time(),
        'repo': str(repo),
        'hardware': {
            'cpu': cpu,
            'ryzen9_detected': 'RYZEN 9' in cpu.upper(),
            'logical_cpus': os.cpu_count(),
            'ram_bytes': ram,
            'ram_gib': None if ram is None else ram / (1024 ** 3),
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable,
            'platform': platform.platform(),
        },
        'torch': torch_info(),
        'git': git_info(repo),
        'version_files': version_files,
        'required_component_evidence': found,
        'missing_component_groups': missing,
        'calibration': None,
        'pilot': None,
        'r7_4_pass': False,
        'status': '',
    }

    if not report['hardware']['ryzen9_detected']:
        report['status'] = 'FAIL_NOT_RYZEN9'
    elif missing:
        report['status'] = 'FAIL_MISSING_R7_STACK'
    else:
        hook = find_one(repo, [
            'tools/r7_4_project_hook.py',
            'tools/r7_4_calibrate_and_pilot.py',
        ])
        if hook is None:
            report['status'] = 'FAIL_MISSING_PROJECT_R7_4_HOOK'
        else:
            hook_path = repo / hook
            hook_out = out.with_name('R7_4_PROJECT_HOOK_REPORT.json')
            proc = subprocess.run(
                [sys.executable, str(hook_path), '--repo', str(repo), '--out', str(hook_out)],
                text=True,
            )
            report['project_hook'] = {
                'path': hook,
                'exit_code': proc.returncode,
                'report': str(hook_out),
            }
            if hook_out.exists():
                try:
                    hook_report = json.loads(hook_out.read_text(encoding='utf-8'))
                    report['calibration'] = hook_report.get('calibration')
                    report['pilot'] = hook_report.get('pilot')
                    report['r7_4_pass'] = bool(hook_report.get('r7_4_pass', False)) and proc.returncode == 0
                except Exception as exc:
                    report['project_hook']['parse_error'] = repr(exc)
            report['status'] = 'PASS' if report['r7_4_pass'] else 'FAIL_PROJECT_HOOK'

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': report['status'],
        'r7_4_pass': report['r7_4_pass'],
        'cpu': cpu,
        'missing_component_groups': missing,
        'report': str(out),
    }, ensure_ascii=False, indent=2))

    return 0 if report['r7_4_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
