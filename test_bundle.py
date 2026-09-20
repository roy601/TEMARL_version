"""Focused regression checks for the portable experiment and paper equations."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import _frozen
from encoders_marl import ARMS, ChoGRUCell, LSTM1997Cell, arm_specs, build_encoder
from env_marl import EpisodeSource, GLOBAL_DIM, LOCAL_DIM, N_AGENTS
from mappo import Actor, critic_input, critic_dim, HP, ValueNorm
from pretrain_marl import HistoryBank, windows
from scripts_marl import verified_runs, loso_folds
from vocab_v2 import PAD_ID


class BundleTests(unittest.TestCase):
    def test_frozen_files(self):
        self.assertTrue(all(_frozen.verify().values()))
        self.assertIn('thesis_system/data/camlds_grounding_verified.json', _frozen.FROZEN_SHA256)

    def test_verified_counts(self):
        runs = verified_runs()
        self.assertEqual(len(runs), 7)
        self.assertEqual(sum(len(v) for v in runs.values()), 36)
        self.assertEqual(sum(len(s) for v in runs.values() for s in v), 1347)

    def test_loso_exclusion(self):
        for fold in loso_folds():
            specs = EpisodeSource(32, fold['train']).take(50)
            self.assertNotIn(fold['held_out'], {s.script for s in specs})

    def test_cho_equation(self):
        cell = ChoGRUCell(4,3).double()
        x,h = torch.randn(2,4,dtype=torch.double),torch.randn(2,3,dtype=torch.double)
        wr,wz,wn = cell.x.weight.chunk(3)
        br,bz,bn = cell.x.bias.chunk(3)
        ur,uz = cell.h_gates.weight.chunk(2)
        r = torch.sigmoid(x@wr.T+br+h@ur.T)
        z = torch.sigmoid(x@wz.T+bz+h@uz.T)
        expected = z*h+(1-z)*torch.tanh(x@wn.T+bn+(r*h)@cell.h_candidate.weight.T)
        torch.testing.assert_close(cell(x,h), expected)

    def test_lstm_1997_equation(self):
        cell = LSTM1997Cell(4,3).double()
        x,h,c = torch.randn(2,4,dtype=torch.double),torch.randn(2,3,dtype=torch.double),torch.randn(2,3,dtype=torch.double)
        gates = F.linear(x,cell.x.weight,cell.x.bias)+F.linear(h,cell.h.weight)
        i,o,g = gates.chunk(3,-1)
        cn = c+i.sigmoid()*(4*g.sigmoid()-2)
        hn = o.sigmoid()*(2*cn.sigmoid()-1)
        actual = cell(x,(h,c))
        torch.testing.assert_close(actual[0],hn)
        torch.testing.assert_close(actual[1],cn)

    def test_capacity_search_preserves_rng(self):
        arm_specs.cache_clear()
        state = torch.random.get_rng_state()
        arm_specs()
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_shapes_masks_gradients_and_checkpoint(self):
        target = arm_specs()['Transformer']['params']
        for arm in ARMS[:3]:
            enc = build_encoder(arm).eval()
            count = sum(p.numel() for p in enc.trunk_parameters())
            self.assertEqual(count,arm_specs()[arm]['params'])
            self.assertLess(abs(count-target)/target,.1)
            prefix = torch.tensor([[1,2,3]])
            longer = torch.tensor([[1,2,3]+[PAD_ID]*13])
            a = enc(prefix)
            b = enc(longer)
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            dirty = longer.clone(); dirty[:,3:] = 11
            torch.testing.assert_close(enc(dirty,[3]),b)
            self.assertTrue(torch.equal(enc(torch.full((1,16),PAD_ID)),torch.zeros(1,64)))
            loss = enc.pretrain_forward(prefix)[1].square().mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None))
            clone = build_encoder(arm).eval()
            clone.load_state_dict(enc.state_dict())
            torch.testing.assert_close(clone(prefix),a)
            enc.freeze()
            self.assertFalse(any(p.requires_grad for p in enc.parameters()))

    def test_history_has_no_future(self):
        a = EpisodeSource(7).next()
        import copy
        b = copy.deepcopy(a)
        b.tau[3:] = (b.tau[3:]+1)%84
        for arm in ARMS:
            enc = build_encoder(arm)
            bank = HistoryBank(None if enc is None else enc.freeze())
            x,y = bank.tables([a,b])
            np.testing.assert_allclose(x[:4],y[:4],atol=1e-6)
            np.testing.assert_array_equal(x[0],np.zeros(64))
        x,l,y = windows([a])
        np.testing.assert_array_equal(y,a.tau[1:])
        self.assertEqual(x[0,0],a.tau[0])

    def test_ippo_critic_locality(self):
        local = np.zeros((2,N_AGENTS,LOCAL_DIM),np.float32)
        h = np.zeros((2,64),np.float32)
        g = np.zeros((2,GLOBAL_DIM),np.float32)
        a = critic_input(False,local,h,g)
        b = critic_input(False,local,h,g+1)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(critic_input(True,local,h,g),critic_input(True,local,h,g+1)))
        self.assertEqual(a.shape[-1],critic_dim(False))
        self.assertEqual(critic_input(True,local,h,g).shape[-1],critic_dim(True))

    def test_value_norm_roundtrip(self):
        vn = ValueNorm()
        x = torch.tensor([1.,2.,9.])
        vn.update(x)
        np.testing.assert_allclose(vn.denormalize(vn.normalize(x.numpy())),x.numpy(),atol=1e-6)

    def test_actor_ppo_identity(self):
        actor = Actor()
        x,h = torch.randn(8,LOCAL_DIM),torch.randn(8,64)
        actor.eval(); old = actor(x,h)[0].detach()
        actor.train(); new = actor(x,h)[0]
        torch.testing.assert_close(old,new)

    def test_six_combinations_and_controls(self):
        from run_experiment import cell_plan
        import experiment
        cells = cell_plan('main',[0])
        self.assertEqual(len(cells),8)
        self.assertEqual(len(experiment.MAIN_ARMS),6)
        self.assertIn(('NoHistory','IPPO'),experiment.CONTROL_ARMS)
        self.assertEqual(len(cell_plan('loso',[0])),56)

    def test_report_uses_validation_not_test(self):
        from run_experiment import cell_plan,cell_name,write_json
        from report_results import generate
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = cell_plan('main',[0])
            write_json(root/'manifest.json',dict(expected_cells=plan, fingerprint='test',smoke=True,study='main',n_test=1))
            for cell in plan:
                trial = dict(script='S1',dwell=100. if cell['encoder']=='GRU' else 1.,depth=1.,protected=0.)
                write_json(root/'cells'/(cell_name(cell)+'.json'),dict(cell=cell,fingerprint='test',smoke=True,
                    val_best=3. if cell['encoder']=='LSTM' else 1.,test=[trial],test_summary=trial))
            generate(root)
            self.assertEqual(json.loads((root/'selection.json').read_text())['encoder'],'LSTM')
            self.assertIn('SMOKE CHECK ONLY',(root/'RESULTS.md').read_text())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
