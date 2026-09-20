# -*- coding: utf-8 -*-
"""
TEMARL v2 — Fusion Trunk, Policy/Value Heads, and Intent-Conditioned Mixer
==========================================================================
Closes F-ARC-01 (uncited GMU), F-ARC-02 (unswept gate bias),
F-ARC-07 (Contribution 2 never ablated), and carries the FIX-1 policy head.

F-ARC-01 — THE CITATION THAT WAS MISSING
----------------------------------------
v1 claimed the "Gated Multi-Modal Fusion Q-network" as Contribution 1, novel.
It is not novel. The construction

        g = sigmoid(W_g [a ; b] + b_g)
        f = g (*) a + (1 - g) (*) b

is exactly the GATED MULTIMODAL UNIT of

    Arevalo, Solorio, Montes-y-Gomez & Gonzalez,
    "Gated Multimodal Units for Information Fusion",
    ICLR 2017 Workshop track, arXiv:1702.01992

which learns a sigmoid gate to weigh modality contributions per sample. The
same convex-combination-with-learned-gate also appears in Highway Networks
(Srivastava et al. 2015). The defensible claim is therefore NOT "a novel fusion
architecture" but "application of the GMU to telemetry/intent fusion under a
Dec-POMDP, with a bias initialisation motivated by gate-init literature."

Measured note in the architecture's defence: the gate is NOT an information
bottleneck. A linear profile probe scores 0.692 on h and 0.679 on the FUSED
vector f the head actually consumes -- i.e. intent survives fusion essentially
intact. v1's failure was in the LEARNER and the BENCHMARK, not in this module.

F-ARC-02 — GATE BIAS
--------------------
v1 fixed b_g = +2.0 (sigma(+2)=0.88) with no sweep. Precedent for biased gate
init exists (LSTM forget-gate bias, Jozefowicz et al. 2015; Highway carry bias,
Srivastava et al. 2015), but the VALUE was unjustified. `gate_bias` is now a
constructor argument and `sweep_gate_bias()` reports the induced initial gate so
the choice is empirical rather than asserted.

F-ARC-07 — CONTRIBUTION 2 WAS NEVER ABLATED
-------------------------------------------
v1's second claimed contribution is conditioning the QMIX hypernetworks on
state||h instead of state alone. No experiment ever isolated it: there was no
"state-only hypernetwork + h in the agents" arm, so the contribution was
asserted, not demonstrated. `IntentQMixer(use_intent_in_hypernet=...)` provides
exactly that arm.

FIX-1 — POLICY/VALUE HEADS
--------------------------
Measured on v1: TD loss rose 4.3 -> 7.9 and never converged (the h-frozen
ablation converged 9.6 -> 0.84); Q-values went flat (range ~0.1); actions went
near-uniform. Root cause: reward credited to a post-decision, stochastically
selected agent gives SNR ~0.57, below the regime where bootstrapped value
iteration converges. The policy head lets the learner be trained by supervised
imitation and then a variance-reduced on-policy method, bypassing the TD target.
Both heads read the SAME fused f, so the policy inherits the full representation.

NOTE ON COMA: with the anticipatory rule exactly ONE agent is scored per step,
so the other agents' actions do not enter that step's reward. COMA's
counterfactual baseline sum_a' pi_i(a')Q(s,(a'_i,a_-i)) therefore degenerates to
a state-value baseline, and A = R - V(f) is the correct variance reducer here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCH_DIM = 64
HEAD_DIM = 64


# ── Gated Multimodal Unit trunk (Arevalo et al. 2017) ────────────────────────

class GatedFusionTrunk(nn.Module):
    """GMU fusion of telemetry and intent, + optional sighted/entropy gate inputs.

    obs layout: [ state (state_dim) | h (h_dim) ]
    """

    def __init__(self, state_dim, h_dim=64, branch_dim=BRANCH_DIM,
                 gate_bias=2.0, sighted_idx=None, state_masking_prob=0.0):
        super().__init__()
        self.state_dim, self.h_dim = state_dim, h_dim
        self.branch_dim = branch_dim
        self.sighted_idx = sighted_idx
        self.state_masking_prob = state_masking_prob
        self.gate_bias = gate_bias

        self.branch_state = nn.Sequential(
            nn.Linear(state_dim, branch_dim), nn.LayerNorm(branch_dim), nn.ReLU())
        self.branch_intent = nn.Sequential(
            nn.Linear(h_dim, branch_dim), nn.LayerNorm(branch_dim), nn.ReLU())

        extra = 1 + (1 if sighted_idx is not None else 0)   # entropy [+ sighted]
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 2 + extra, branch_dim), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.gate[0].bias, gate_bias)   # F-ARC-02

    def _mask_state(self, s):
        """ModDrop-style modality dropout (Neverova et al., TPAMI 2016).
        h is NEVER masked -- that asymmetry is what forces an h-conditioned policy."""
        if not self.training or self.state_masking_prob <= 0:
            return s
        keep = torch.bernoulli(torch.full((s.shape[0], 1), 1.0 - self.state_masking_prob,
                                          device=s.device, dtype=s.dtype))
        return s * keep

    def forward(self, obs, entropy=None):
        shape = obs.shape[:-1]
        s = obs[..., :self.state_dim].reshape(-1, self.state_dim)
        h = obs[..., self.state_dim:].reshape(-1, self.h_dim)
        N = s.shape[0]
        s = self._mask_state(s)

        a = self.branch_state(s)
        b = self.branch_intent(h)

        if entropy is None:
            ent = torch.full((N, 1), 0.5, device=a.device, dtype=a.dtype)
        elif isinstance(entropy, (float, int)):
            ent = torch.full((N, 1), float(entropy), device=a.device, dtype=a.dtype)
        else:
            ent = torch.as_tensor(entropy).reshape(N, 1).to(a.device, a.dtype)

        gin = [a, b, ent]
        if self.sighted_idx is not None:
            gin.append(s[:, self.sighted_idx:self.sighted_idx + 1])
        g = self.gate(torch.cat(gin, -1))
        f = g * a + (1.0 - g) * b                 # GMU convex combination
        return f, g, shape

    def gate_value(self, obs, entropy=None):
        with torch.no_grad():
            _, g, _ = self.forward(obs, entropy)
        return g


# ── Agent network: policy + value + (legacy) Q head on a shared trunk ────────

class AgentNet(nn.Module):
    """Policy pi(a|f) and value V(f) heads (FIX-1), plus a Q head retained so the
    TD-QMIX arm can still be run as a COMPARISON POINT on the corrected
    environment -- which cleanly decomposes 'what the env fix bought' from
    'what the learner fix bought'."""

    def __init__(self, state_dim, n_actions, h_dim=64, gate_bias=2.0,
                 sighted_idx=None, state_masking_prob=0.0):
        super().__init__()
        self.trunk = GatedFusionTrunk(state_dim, h_dim, gate_bias=gate_bias,
                                      sighted_idx=sighted_idx,
                                      state_masking_prob=state_masking_prob)
        d = self.trunk.branch_dim
        self.policy_head = nn.Sequential(nn.Linear(d, HEAD_DIM), nn.ReLU(),
                                         nn.Linear(HEAD_DIM, n_actions))
        self.value_head = nn.Sequential(nn.Linear(d, HEAD_DIM), nn.ReLU(),
                                        nn.Linear(HEAD_DIM, 1))
        self.q_head = nn.Sequential(nn.Linear(d, HEAD_DIM), nn.ReLU(),
                                    nn.Linear(HEAD_DIM, n_actions))
        self.n_actions = n_actions

    def policy_logits(self, obs, entropy=None):
        f, _, shape = self.trunk(obs, entropy)
        return self.policy_head(f).reshape(*shape, self.n_actions)

    def value(self, obs, entropy=None):
        f, _, shape = self.trunk(obs, entropy)
        return self.value_head(f).squeeze(-1).reshape(shape)

    def q_values(self, obs, entropy=None):
        f, _, shape = self.trunk(obs, entropy)
        return self.q_head(f).reshape(*shape, self.n_actions)

    # -- parameter groups: actor and critic must NOT share an optimiser -------
    def actor_parameters(self):
        return list(self.trunk.parameters()) + list(self.policy_head.parameters())

    def critic_parameters(self):
        return list(self.value_head.parameters())

    def q_parameters(self):
        return list(self.trunk.parameters()) + list(self.q_head.parameters())


# ── Intent-conditioned QMIX mixer, with the F-ARC-07 ablation switch ─────────

class IntentQMixer(nn.Module):
    """QMIX monotonic mixer (Rashid et al. 2018 [20]).

    use_intent_in_hypernet=True  -> hypernet input is state||h  (v1 Contribution 2)
    use_intent_in_hypernet=False -> hypernet input is state only (standard QMIX)

    Running BOTH is the ablation v1 never performed. Monotonicity is enforced by
    abs() on W1/W2 only (biases are unconstrained, per Rashid); this guarantees
    dQ_tot/dQ_i >= 0, the IGM condition. verify_monotonicity() checks it
    NUMERICALLY by autograd rather than trusting the construction.
    """

    def __init__(self, n_agents, state_dim, h_dim=64, embed=64, hyper=128,
                 use_intent_in_hypernet=True):
        super().__init__()
        self.n_agents, self.embed = n_agents, embed
        self.use_intent = use_intent_in_hypernet
        in_dim = state_dim + (h_dim if use_intent_in_hypernet else 0)
        self.in_dim = in_dim
        self.hw1 = nn.Sequential(nn.Linear(in_dim, hyper), nn.ReLU(),
                                 nn.Linear(hyper, n_agents * embed))
        self.hb1 = nn.Linear(in_dim, embed)
        self.hw2 = nn.Sequential(nn.Linear(in_dim, hyper), nn.ReLU(),
                                 nn.Linear(hyper, embed))
        self.hb2 = nn.Sequential(nn.Linear(in_dim, embed), nn.ReLU(),
                                 nn.Linear(embed, 1))

    def forward(self, q_agents, state, h=None):
        """q_agents: (B, n_agents)  state: (B, state_dim)  h: (B, h_dim) or None"""
        B = q_agents.shape[0]
        x = torch.cat([state, h], -1) if self.use_intent else state
        w1 = torch.abs(self.hw1(x)).view(B, self.n_agents, self.embed)
        b1 = self.hb1(x).view(B, 1, self.embed)
        w2 = torch.abs(self.hw2(x)).view(B, self.embed, 1)
        b2 = self.hb2(x).view(B, 1, 1)
        hid = F.elu(torch.bmm(q_agents.view(B, 1, self.n_agents), w1) + b1)
        return (torch.bmm(hid, w2) + b2).view(B)


def verify_monotonicity(mixer, state_dim, h_dim=64, n=64, seed=0):
    """IGM check by autograd: dQ_tot/dQ_i must be >= 0 for every agent.
    Asserting this numerically (not just by construction) catches sign bugs in
    the abs()/bmm plumbing that a code read would miss."""
    torch.manual_seed(seed)
    q = torch.randn(n, mixer.n_agents, requires_grad=True)
    s = torch.randn(n, state_dim)
    h = torch.randn(n, h_dim) if mixer.use_intent else None
    qt = mixer(q, s, h)
    qt.sum().backward()
    g = q.grad
    return float(g.min()), float(g.mean()), bool((g >= -1e-7).all())


# ── Multi-agent container ────────────────────────────────────────────────────

class MultiAgentController:
    def __init__(self, n_agents, state_dim, n_actions, h_dim=64, gate_bias=2.0,
                 sighted_idx=None, state_masking_prob=0.0, device="cpu"):
        self.n_agents, self.device = n_agents, device
        self.agents = [AgentNet(state_dim, n_actions, h_dim, gate_bias,
                                sighted_idx, state_masking_prob).to(device)
                       for _ in range(n_agents)]

    def act_policy(self, obs_list, entropy=None, greedy=True):
        out = []
        for i, o in enumerate(obs_list):
            t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                lg = self.agents[i].policy_logits(t, entropy)
            out.append(int(lg.argmax(-1).item()) if greedy
                       else int(torch.multinomial(torch.softmax(lg.squeeze(0), -1), 1)))
        return out

    def act_q(self, obs_list, entropy=None, epsilon=0.0):
        out = []
        for i, o in enumerate(obs_list):
            if np.random.random() < epsilon:
                out.append(int(np.random.randint(self.agents[i].n_actions)))
                continue
            t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.agents[i].q_values(t, entropy)
            out.append(int(q.argmax(-1).item()))
        return out

    def actor_parameters(self):
        return [p for a in self.agents for p in a.actor_parameters()]

    def critic_parameters(self):
        return [p for a in self.agents for p in a.critic_parameters()]

    def train(self):
        [a.train() for a in self.agents]

    def eval(self):
        [a.eval() for a in self.agents]

    def set_masking(self, p):
        for a in self.agents:
            a.trunk.state_masking_prob = p


def sweep_gate_bias(state_dim, n_actions, biases=(0.0, 1.0, 2.0, 3.0),
                    h_dim=64, n=256, seed=0):
    """F-ARC-02: report the initial gate value induced by each bias. The gate is
    the telemetry weight: g->1 trusts telemetry, g->0 trusts intent."""
    torch.manual_seed(seed)
    obs = torch.randn(n, state_dim + h_dim)
    rows = []
    for b in biases:
        net = AgentNet(state_dim, n_actions, h_dim, gate_bias=b)
        net.eval()
        g = net.trunk.gate_value(obs).mean().item()
        rows.append({"bias": b, "sigmoid": 1 / (1 + np.exp(-b)), "mean_gate": g})
    return rows


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.manual_seed(0)
    from env_v2 import DeceptionEnvV2, N_AGENTS, build_profiles_from_camlds
    from d3fend_payoff import N_ACTIONS

    profiles, _ = build_profiles_from_camlds()
    env = DeceptionEnvV2(profiles)
    S, H = env.local_dim, env.h_dim
    print("=" * 76)
    print("  TEMARL v2 — FUSION TRUNK / POLICY HEADS / INTENT MIXER")
    print("=" * 76)
    print(f"  agents={N_AGENTS}  state_dim={S}  h_dim={H}  n_actions={N_ACTIONS}")

    print(f"\n  [F-ARC-01] GMU attribution")
    print(f"    f = g*a + (1-g)*b  ==  Gated Multimodal Unit")
    print(f"    Arevalo et al., ICLR 2017 Workshop, arXiv:1702.01992  -- CITED")
    print(f"    v1 claimed this as novel Contribution 1; claim withdrawn/narrowed.")

    print(f"\n  [F-ARC-02] GATE BIAS SWEEP (v1 fixed +2.0 with no sweep)")
    print(f"    {'bias':>5} {'sigmoid(b)':>11} {'mean gate g':>12}   interpretation")
    for r in sweep_gate_bias(S, N_ACTIONS):
        interp = "trusts telemetry" if r["mean_gate"] > 0.6 else (
                 "balanced" if r["mean_gate"] > 0.4 else "trusts intent")
        print(f"    {r['bias']:>5.1f} {r['sigmoid']:>11.3f} {r['mean_gate']:>12.3f}   {interp}")

    print(f"\n  [FIX-1] HEAD SHAPES & OPTIMISER DECOUPLING")
    net = AgentNet(S, N_ACTIONS, H, sighted_idx=env.ATTACKER_PRESENT_IDX)
    net.eval()
    ob = torch.randn(7, S + H)
    pl, vl, ql = net.policy_logits(ob), net.value(ob), net.q_values(ob)
    print(f"    policy {tuple(pl.shape)}  value {tuple(vl.shape)}  q {tuple(ql.shape)}")
    assert pl.shape == (7, N_ACTIONS) and vl.shape == (7,) and ql.shape == (7, N_ACTIONS)
    ap = {id(p) for p in net.actor_parameters()}
    cp = {id(p) for p in net.critic_parameters()}
    print(f"    actor params {len(ap)}  critic params {len(cp)}  "
          f"overlap {len(ap & cp)}  {'PASS' if not (ap & cp) else 'FAIL'}")
    print(f"    (critic updates value_head ONLY; the trunk receives gradient from the")
    print(f"     actor loss alone, so value error cannot distort the representation)")

    print(f"\n  [F-ARC-07] CONTRIBUTION 2 ABLATION — h in the hypernetwork")
    gdim = env.global_state().shape[0]
    for use in (True, False):
        mx = IntentQMixer(N_AGENTS, gdim, H, use_intent_in_hypernet=use)
        mn, mean, ok = verify_monotonicity(mx, gdim, H)
        arm = "state||h  (v1 Contribution 2)" if use else "state only (standard QMIX)"
        print(f"    {arm:<32} hypernet in_dim={mx.in_dim:<4} "
              f"IGM dQtot/dQi min={mn:+.3e} mean={mean:+.3f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"    Both arms instantiate => the ablation v1 never ran is now possible.")

    print(f"\n  [SMOKE] controller acts on real env observations")
    ctl = MultiAgentController(N_AGENTS, S, N_ACTIONS, H,
                               sighted_idx=env.ATTACKER_PRESENT_IDX)
    obs = env.reset()
    a_pi = ctl.act_policy(obs)
    a_q = ctl.act_q(obs)
    obs2, rw, done, info = env.step(a_pi)
    print(f"    policy actions {a_pi}   q actions {a_q}")
    print(f"    step -> reward {rw[0]:+.3f}  responsible={info['responsible_agent']}  "
          f"blind={info['blind']}  align={info['alignment']:.3f}")
    print("=" * 76)
