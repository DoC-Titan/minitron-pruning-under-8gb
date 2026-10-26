"""Focused math, structural, update and interruption tests; standard unittest."""
import tempfile
import unittest
from unittest import mock
from transformers import Qwen2Config, Qwen2ForCausalLM
from core import *


def tiny_model(seed=42):
    seed_all(seed)
    config=Qwen2Config(vocab_size=256,hidden_size=64,intermediate_size=192,num_hidden_layers=4,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,
                       tie_word_embeddings=True,attention_dropout=0.)
    config._attn_implementation='sdpa'
    return Qwen2ForCausalLM(config)


class CorrectnessTests(unittest.TestCase):
    def test_committed_shard_missing_receipt_is_repaired(self):
        import worker
        job=jobs()[0]
        with tempfile.TemporaryDirectory(dir=ROOT/'state') as d:
            root=Path(d);(root/'config.json').write_text((ROOT/'config.json').read_text())
            recipe=root/'recipe.json';recipe.write_text('{}')
            state_path=root/'state'/job['id']/'resume.pt';state_path.parent.mkdir(parents=True);state_path.write_bytes(b'placeholder')
            state={'job':job,'config_sha256':sha(root/'config.json'),'recipe_sha256':sha(recipe),
                   'next_shard':1,'totals':{'input_tokens':8192},'recovery':[]}
            with mock.patch.object(worker,'ROOT',root), mock.patch.object(worker,'recipe_path',return_value=recipe), \
                 mock.patch.object(worker,'checked_load',return_value=state), mock.patch.object(worker,'construct',side_effect=AssertionError('Must not retrain')):
                worker.train_shard(job,0)
            receipt=json.loads((state_path.parent/'shard_000.done.json').read_text())
            self.assertTrue(receipt['receipt_recovered_from_checkpoint']);self.assertEqual(receipt['next_shard'],1)

    def test_kl_direction_temperature_and_gradient(self):
        s=torch.tensor([[1.,-1.,0.],[.2,.3,.7]],requires_grad=True)
        t=torch.tensor([[.1,.5,.2],[2.,0.,-1.]])
        p=(t/2).softmax(-1); expected=(p*(p.log()-(s/2).log_softmax(-1))).sum(-1).mean()*4
        actual=full_kl(s,t,2);self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        actual.backward();self.assertTrue(torch.isfinite(s.grad).all());self.assertGreater(float(s.grad.abs().sum()),0)
        self.assertLess(abs(float(full_kl(t,t,2))),1e-6)

    def test_physical_shapes_and_round_trip(self):
        m=tiny_model();before=sum(p.numel() for p in m.parameters())
        recipe={'retained_layers':[0,2,3],'ffn':128,'channels':[{'retained':list(range(128))} for _ in range(3)]}
        apply_recipe(m,recipe);self.assertLess(sum(p.numel() for p in m.parameters()),before)
        x=torch.randint(0,256,(1,16));m.eval()
        with torch.no_grad():z=m(input_ids=x,use_cache=False).logits
        with tempfile.TemporaryDirectory(dir=ROOT/'state') as d:
            m.save_pretrained(d);r=Qwen2ForCausalLM.from_pretrained(d);r.eval()
            with torch.no_grad():self.assertTrue(torch.allclose(z,r(input_ids=x,use_cache=False).logits,atol=1e-6))
        self.assertEqual(m.config.num_hidden_layers,3);self.assertEqual(m.config.intermediate_size,128)

    def test_resume_equivalence_and_hash_fallback(self):
        def build():
            m=tiny_model();m.requires_grad_(False);add_adapters(m)
            return m,torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        m,o=build();x=torch.arange(16).reshape(1,-1)
        def step(m,o):
            m.train();o.zero_grad();z=m(input_ids=x,use_cache=False).logits[0,:-1]
            F.cross_entropy(z,x[0,1:]).backward();o.step()
        step(m,o)
        with tempfile.TemporaryDirectory(dir=ROOT/'state') as d:
            path=Path(d)/'resume.pt';value={'adapter':adapter_state(m),'optimizer':o.state_dict()}
            atomic_torch(path,value);atomic_torch(path,value,backup=True)
            # Corrupted current file must fall back to the last verified checkpoint.
            path.write_bytes(b'interrupted');saved=checked_load(path,backup=True)
            atomic_torch(path,saved,backup=True)
            path.write_bytes(b'second interruption')
            saved=checked_load(path,backup=True)
            r,ro=build();r.load_state_dict(saved['adapter'],strict=False);ro.load_state_dict(saved['optimizer'])
            step(m,o);step(r,ro)
            self.assertTrue(all(torch.equal(v,adapter_state(r)[k]) for k,v in adapter_state(m).items()))


if __name__=='__main__':
    (ROOT/'state').mkdir(exist_ok=True);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CorrectnessTests))
    atomic_json(ROOT/'state/tests.done.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors)})
    raise SystemExit(0 if result.wasSuccessful() else 1)
