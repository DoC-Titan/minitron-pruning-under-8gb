"""Device-explicit primitives. No downloads, legacy imports, or writes on import."""
import contextlib
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / 'config.json').read_text())
GIB = 2 ** 30
TEACHER = (ROOT / os.environ.get('LEVEL4_TEACHER', C['teacher_relative_path'])).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def tensor_hash(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, value, backup=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    checksum = sha(tmp)
    valid_current = False
    if backup and path.exists():
        try:
            valid_current = sha(path) == json.loads(path.with_suffix('.sha256').read_text())['sha256']
        except (OSError, ValueError, KeyError):
            pass
    # Never overwrite a good previous checkpoint with an already-corrupted current.
    if backup and valid_current:
        os.replace(path, path.with_suffix('.previous.pt'))
        check = path.with_suffix('.sha256')
        if check.exists(): os.replace(check, path.with_suffix('.previous.sha256'))
    os.replace(tmp, path)
    # A power failure between data/hash rename is detected; previous is recoverable.
    atomic_json(path.with_suffix('.sha256'), {'sha256': checksum})


def checked_load(path, backup=False):
    path = Path(path)
    candidates = [path, path.with_suffix('.previous.pt')] if backup else [path]
    errors = []
    for candidate in candidates:
        try:
            expected = json.loads(candidate.with_suffix('.sha256').read_text())['sha256']
            assert sha(candidate) == expected, 'Checkpoint hash mismatch'
            return torch.load(candidate, map_location='cpu', weights_only=True)
        except (OSError, AssertionError, ValueError, RuntimeError, EOFError) as exc:
            errors.append(str(exc))
    raise RuntimeError('No valid checkpoint: ' + '; '.join(errors))


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n'); f.flush()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(C['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def gpu_sample():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()[0]
        a = [int(x.strip()) for x in out.split(',')]
        return dict(temperature_c=a[0], memory_used_mib=a[1], memory_total_mib=a[2], utilization_percent=a[3])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'monitor_error': str(exc)}


def guard(device='cpu', check_size=False):
    s = C['safety']
    if shutil.disk_usage(ROOT).free < s['min_free_disk_gib'] * GIB: raise RuntimeError('Safety stop: free disk')
    if psutil.virtual_memory().available < s['min_available_ram_gib'] * GIB: raise RuntimeError('Safety stop: available RAM')
    if check_size:
        size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if size > s['max_extension_gib'] * GIB: raise RuntimeError('Safety stop: extension disk ceiling')
    if str(device).startswith('cuda'):
        sample = gpu_sample()
        if 'monitor_error' in sample: raise RuntimeError('GPU monitor unavailable: ' + sample['monitor_error'])
        if sample['temperature_c'] >= s['gpu_pause_celsius']:
            start = time.monotonic()
            while sample['temperature_c'] > s['gpu_resume_celsius']:
                if time.monotonic() - start > s['max_cooldown_seconds']: raise RuntimeError('Safety stop: temperature')
                time.sleep(15); sample = gpu_sample()
                if 'monitor_error' in sample: raise RuntimeError('Temperature monitor failed')
        return sample
    return {'rss_gib': psutil.Process().memory_info().rss / GIB, 'available_ram_gib': psutil.virtual_memory().available / GIB}


def configure(device):
    seed_all(42)
    if device == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA explicitly requested but unavailable; no silent fallback')
        torch.cuda.set_per_process_memory_fraction(C['safety']['cuda_memory_fraction'])
    guard(device)


def memory(device):
    result = {'rss_gib': psutil.Process().memory_info().rss / GIB,
              'process_peak_working_set_gib': getattr(psutil.Process().memory_info(), 'peak_wset', 0) / GIB}
    if str(device).startswith('cuda'):
        result.update(allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                      reserved_gib=torch.cuda.max_memory_reserved() / GIB, system_gpu=gpu_sample())
    return result


def load_model(device='cuda', path=TEACHER, dtype=None):
    guard(device)
    dtype = dtype or (torch.float16 if device == 'cuda' else torch.float32)
    start = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=dtype,
            device_map={'': device}, low_cpu_mem_usage=True, attn_implementation='sdpa')
    m.requires_grad_(False); m.eval(); sync(device)
    assert all(p.device.type == device for p in m.parameters()), 'Mixed/fallback placement'
    return m, time.perf_counter() - start


def tokenizer(): return AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)


def fix_layers(model):
    model.config.num_hidden_layers = len(model.model.layers)
    model.config.max_window_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers): layer.self_attn.layer_idx = i


def sliced_linear(old, ids, dim):
    w = old.weight.detach().index_select(dim, ids.to(old.weight.device)).contiguous()
    new = nn.Linear(w.shape[1], w.shape[0], bias=old.bias is not None, device=w.device, dtype=w.dtype)
    new.weight = nn.Parameter(w, requires_grad=False)
    if old.bias is not None:
        b = old.bias.detach().index_select(0, ids.to(w.device)) if dim == 0 else old.bias.detach().clone()
        new.bias = nn.Parameter(b, requires_grad=False)
    return new


def apply_recipe(model, recipe):
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in recipe['retained_layers']])
    fix_layers(model)
    for layer, sel in zip(model.model.layers, recipe['channels']):
        mlp = layer.mlp; ids = torch.tensor(sel['retained'], dtype=torch.long)
        mlp.gate_proj = sliced_linear(mlp.gate_proj, ids, 0)
        mlp.up_proj = sliced_linear(mlp.up_proj, ids, 0)
        mlp.down_proj = sliced_linear(mlp.down_proj, ids, 1)
        mlp.intermediate_size = len(ids)
        assert mlp.gate_proj.out_features == mlp.up_proj.out_features == mlp.down_proj.in_features == len(ids)
    model.config.intermediate_size = recipe['ffn']
    gc.collect()
    if next(model.parameters()).is_cuda: torch.cuda.empty_cache()
    return model


class LoRALinear(nn.Module):
    def __init__(self, base, rank=None, alpha=None):
        super().__init__(); self.base = base
        rank = rank or C['rank']; self.scale = (alpha or C['alpha']) / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        return self.base(x) + delta.to(x.dtype)

    def merge(self):
        with torch.no_grad(): self.base.weight.add_((self.B @ self.A * self.scale).to(self.base.weight.dtype))
        return self.base


def add_adapters(model):
    for layer in model.model.layers:
        for key in ['q_proj', 'v_proj']: setattr(layer.self_attn, key, LoRALinear(getattr(layer.self_attn, key)))


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.endswith(('.A', '.B'))}


def full_kl(student, teacher, temperature=None):
    t = temperature or C['temperature']
    s = F.log_softmax(student.float() / t, -1)
    p = F.log_softmax(teacher.detach().float() / t, -1)
    return F.kl_div(s, p, log_target=True, reduction='none').sum(-1).mean() * t * t


@torch.inference_mode()
def nll(model, blocks, device):
    start = time.perf_counter(); values = []; tokens = 0
    model.eval()
    for i, row in enumerate(blocks):
        if i % 32 == 0: guard(device)
        x = row.unsqueeze(0).to(device)
        assert x.device == next(model.parameters()).device
        logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction='sum')
        assert torch.isfinite(loss)
        values.append(float(loss) / (x.shape[1] - 1)); tokens += x.shape[1] - 1
    sync(device)
    mean = statistics.mean(values)
    return {'nll': mean, 'perplexity': math.exp(mean), 'scored_tokens': tokens,
            'per_block_nll': values, 'seconds': time.perf_counter() - start}


@torch.inference_mode()
def piqa(model, tok, records, device):
    start = time.perf_counter(); rows = []; model.eval()
    for i, rec in enumerate(records):
        if i % 32 == 0: guard(device)
        prefix = tok('Question: ' + rec['goal'] + '\nAnswer:', add_special_tokens=False)['input_ids']
        scores = []
        for key in ['sol1', 'sol2']:
            answer = tok(' ' + rec[key], add_special_tokens=False)['input_ids']
            x = torch.tensor([prefix + answer], device=device)
            z = model(input_ids=x, use_cache=False).logits[:, len(prefix)-1:-1].float()
            scores.append(-float(F.cross_entropy(z.reshape(-1, z.shape[-1]), torch.tensor(answer, device=device))))
        rows.append({'index': rec['original_index'], 'label': rec['label'], 'prediction': int(scores[1] > scores[0]), 'scores': scores})
    return {'accuracy': sum(x['label'] == x['prediction'] for x in rows) / len(rows), 'count': len(rows),
            'details': rows, 'seconds': time.perf_counter()-start,
            'method': 'custom zero-shot length-normalized answer conditional likelihood; not official lm-eval harness'}


def jobs():
    return [{'id': f'{level}_{criterion}_s{seed}_{objective}', 'level': level, 'criterion': criterion,
             'seed': seed, 'objective': objective}
            for level in C['structures'] for criterion in C['criteria'] for seed in C['seeds']
            for objective in (['kl', 'ce'] if criterion == 'informed' else ['kl'])]


def recipe_path(job): return ROOT / 'recipes' / f'{job["level"]}_{job["criterion"]}_s{job["seed"]}.json'


def environment():
    def run(cmd):
        try: return subprocess.check_output(cmd, text=True, timeout=20).strip()
        except (OSError, subprocess.SubprocessError) as e: return str(e)
    versions = {}
    for package in ['torch', 'transformers', 'numpy', 'datasets', 'accelerate', 'safetensors', 'psutil', 'tensorrt_llm', 'nvidia-modelopt', 'nemo_toolkit', 'cuml']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = 'NOT INSTALLED / NOT USED'
    return {'recorded_at': dt.datetime.now().astimezone().isoformat(), 'platform': platform.platform(),
            'python': sys.version, 'cpu': run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name']),
            'physical_cores': psutil.cpu_count(False), 'logical_cores': psutil.cpu_count(),
            'ram_total_gib': psutil.virtual_memory().total/GIB, 'ram_available_gib': psutil.virtual_memory().available/GIB,
            'cpu_threads': torch.get_num_threads(), 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'gpu_vram_gib': torch.cuda.get_device_properties(0).total_memory/GIB if torch.cuda.is_available() else None,
            'compute_capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            'gpu_driver': run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
            'nvidia_smi': run(['nvidia-smi']), 'packages': versions, 'seeds': C['seeds'],
            'cuml_accel_fallback': 'N/A: no RAPIDS/cuML estimator or accelerator used',
            'cpu_temperature': 'Unavailable: Windows/psutil exposes no verified CPU temperature sensor; not measured',
            'tf32': False, 'determinism': 'fixed RNG seeds, no dropout, TF32 off; backend bitwise determinism not guaranteed',
            'safety': C['safety']}
