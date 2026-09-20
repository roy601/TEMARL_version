"""New study specification; not the historical v7 pre-registration."""
ENCODERS = ("Transformer", "GRU", "LSTM")
LEARNERS = ("IPPO", "MAPPO")
MAIN_ARMS = tuple((e, l) for l in LEARNERS for e in ENCODERS)
# Both controls isolate intent within each learner; the requested three-row
# ablation is emitted as a subset after validation-only MAPPO selection.
CONTROL_ARMS = (("NoHistory", "IPPO"), ("NoHistory", "MAPPO"))
SEEDS = tuple(range(10))
LOSO_SEEDS = tuple(range(5))
ENV_STEPS = 2_000_000
N_TEST = 500
METRICS = ("dwell", "depth", "protected")
SELECTION = "Highest mean best-validation dwell across MAPPO seeds; ties use ENCODERS order."
