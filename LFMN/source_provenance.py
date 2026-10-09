"""Source identity for Git checkouts and checksum-verified offline packages."""
import hashlib
import json
from pathlib import Path
import subprocess


def provenance(root):
    root = Path(root).resolve()
    if (root / '.git').exists():
        def git(*arguments):
            return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()
        return {'git_commit': git('rev-parse', 'HEAD'),
                'git_dirty': bool(git('status', '--porcelain')),
                'git_diff_summary': git('diff', '--stat'), 'source_mode': 'git'}
    path = root / 'OFFLINE_SOURCE.json'
    if not path.exists():
        raise RuntimeError('No Git checkout or OFFLINE_SOURCE.json; use the complete offline package')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError('offline manifest path escapes package: ' + name)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('offline source checksum mismatch: ' + name)
    return {'git_commit': manifest['source_commit'], 'git_dirty': None,
            'git_diff_summary': 'No Git metadata; offline files verified against package manifest',
            'source_mode': 'verified_offline_archive'}
