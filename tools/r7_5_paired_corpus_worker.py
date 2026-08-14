from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import r7_4_stability_pilot_worker as mono
import r7_4_staged_domain_worker as staged
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from spincore.deep_cfr import _validate_policy, icm_delta_utility
from spincore.r7_5_paired_corpus import BottomHashCorpus, PairedSample
from spincore.solver import SolverLibrary

PAYOUT = (0.5, 0.3, 0.2)
SCHEMA = "SPINCORE_R7_5_PAIRED_CORPUS_V1"


class PairedPartialExactCollector(PartialExactAdvantageCollector):
    """R7.4 behavior semantics plus side-effect-free V1/V2 sample pairing."""

    def __init__(
        self,
        *,
        policy,
        terminal_utility,
        rng,
        advantage_memory,
        strategy_memory,
        paired_advantage: BottomHashCorpus[PairedSample],
        paired_strategy: BottomHashCorpus[PairedSample],
        domain: str,
        corpus_seed: int,
    ):
        super().__init__(
            policy=policy,
            terminal_utility=terminal_utility,
            rng=rng,
            advantage_memory=advantage_memory,
            strategy_memory=strategy_memory,
        )
        self.paired_advantage = paired_advantage
        self.paired_strategy = paired_strategy
        self.domain_name = str(domain)
        self.corpus_seed = int(corpus_seed)

    @staticmethod
    def _legal_mask(state) -> tuple[int, ...]:
        legal = set(int(action) for action in state.legal_actions())
        return tuple(1 if action in legal else 0 for action in range(6))

    def _add_pair(self, *, kind: str, state, target, weight: float, iteration: int) -> None:
        sample = PairedSample(
            kind=kind,
            domain=self.domain_name,
            corpus_seed=self.corpus_seed,
            observation_v1=state.neural_bytes(),
            observation_v2=state.neural_bytes_v2(),
            legal=self._legal_mask(state),
            target=tuple(float(value) for value in target),
            weight=float(weight),
            iteration=int(iteration),
        )
        (self.paired_advantage if kind == "advantage" else self.paired_strategy).add(sample)

    def _adv_partial(self, state, traverser, iteration, exact_opponent_levels, exact_depth):
        self.nodes += 1
        if state.terminal:
            return self._utility(state, traverser)
        actor = state.actor
        observation = state.neural_bytes()
        legal = state.legal_actions()
        probabilities = self._p(observation, legal, actor, state.domain)
        if actor == traverser:
            utilities = [0.0] * 6
            node_utility = 0.0
            for action in legal:
                child = state.child(action)
                try:
                    utilities[action] = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_opponent_levels,
                        exact_depth,
                    )
                finally:
                    child.close()
                node_utility += probabilities[action] * utilities[action]
            regrets = [0.0] * 6
            for action in legal:
                regrets[action] = utilities[action] - node_utility
            sample_weight = max(1.0, float(iteration))
            self.advantage_memory.add(
                __import__("spincore.deep_cfr", fromlist=["AdvantageSample"]).AdvantageSample(
                    observation,
                    tuple(regrets),
                    tuple(1 if action in legal else 0 for action in range(6)),
                    sample_weight,
                    state.domain,
                )
            )
            self._add_pair(
                kind="advantage",
                state=state,
                target=regrets,
                weight=sample_weight,
                iteration=iteration,
            )
            self.samples += 1
            return node_utility

        if exact_depth < exact_opponent_levels:
            node_utility = 0.0
            for action in legal:
                probability = probabilities[action]
                if probability <= 0.0:
                    continue
                child = state.child(action)
                try:
                    child_utility = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_opponent_levels,
                        exact_depth + 1,
                    )
                finally:
                    child.close()
                node_utility += probability * child_utility
            return node_utility

        action = self._sample(probabilities, legal)
        child = state.child(action)
        try:
            return self._adv_partial(
                child,
                traverser,
                iteration,
                exact_opponent_levels,
                exact_depth,
            )
        finally:
            child.close()

    def _strategy(self, state, target_player, iteration, own_reach):
        if state.terminal:
            return
        actor = state.actor
        observation = state.neural_bytes()
        legal = state.legal_actions()
        probabilities = self._p(observation, legal, actor, state.domain)
        if actor == target_player:
            sample_weight = max(float(own_reach), 1.0e-12) * max(1.0, float(iteration))
            self.strategy_memory.add(
                __import__("spincore.deep_cfr", fromlist=["StrategySample"]).StrategySample(
                    observation,
                    tuple(probabilities),
                    tuple(1 if action in legal else 0 for action in range(6)),
                    sample_weight,
                    state.domain,
                )
            )
            self._add_pair(
                kind="strategy",
                state=state,
                target=probabilities,
                weight=sample_weight,
                iteration=iteration,
            )
            for action in legal:
                probability = probabilities[action]
                if probability <= 0.0:
                    continue
                child = state.child(action)
                try:
                    self._strategy(child, target_player, iteration, own_reach * probability)
                finally:
                    child.close()
            return
        action = self._sample(probabilities, legal)
        child = state.child(action)
        try:
            self._strategy(child, target_player, iteration, own_reach)
        finally:
            child.close()


def _save_samples(path: Path, samples: list[PairedSample]) -> None:
    payload = [
        {
            "kind": sample.kind,
            "domain": sample.domain,
            "corpus_seed": sample.corpus_seed,
            "observation_v1": sample.observation_v1,
            "observation_v2": sample.observation_v2,
            "legal": sample.legal,
            "target": sample.target,
            "weight": sample.weight,
            "iteration": sample.iteration,
        }
        for sample in samples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _collect_chunk(
    *,
    solver,
    collector: PairedPartialExactCollector,
    bundle,
    seed: int,
    domain: str,
    iteration: int,
    roots: int,
    global_root: int,
    scenario_counts: list[int],
    exact_opponent_levels: int,
) -> int:
    scenarios = mono._scenario_cycle(domain)
    live_by_scenario = [tuple(i for i, stack in enumerate(ep.stacks) if stack > 0) for ep in scenarios]
    for _ in range(int(roots)):
        scenario_index = global_root % len(scenarios)
        episode = scenarios[scenario_index]
        live = live_by_scenario[scenario_index]
        scenario_counts[scenario_index] += 1
        deck_seed = (int(seed) * 1_000_003 + global_root * 97 + int(iteration)) & ((1 << 64) - 1)
        for traverser in live:
            root = solver.create(episode, deck_seed)
            try:
                collector.collect_advantage_partial_exact(
                    root,
                    traverser=int(traverser),
                    iteration=int(iteration),
                    exact_opponent_levels=int(exact_opponent_levels),
                )
            finally:
                root.close()
        for target_player in live:
            root = solver.create(episode, deck_seed)
            try:
                collector.collect_strategy_own_reach(
                    root,
                    target_player=int(target_player),
                    iteration=int(iteration),
                )
            finally:
                root.close()
        global_root += 1
        bundle.counters["roots"] += 1
    return global_root


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 paired SPNNIV1/SPNNIV2 corpus worker")
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--domain", choices=["TRUE_HEADS_UP", "THREE_HANDED"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-roots", type=int, default=128)
    parser.add_argument("--initial-paired-roots", type=int, default=64)
    parser.add_argument("--extension-roots", type=int, default=64)
    parser.add_argument("--max-paired-roots", type=int, default=192)
    parser.add_argument("--min-advantage", type=int, default=5000)
    parser.add_argument("--min-strategy", type=int, default=2000)
    parser.add_argument("--capacity", type=int, default=100000)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    solver = SolverLibrary(args.lib)
    run_args = mono._args_from_freeze(
        freeze,
        roots_per_iteration=int(args.bootstrap_roots),
        device=args.device,
    )
    if int(run_args.exact_opponent_levels) != 2:
        raise ValueError("R7.5 paired corpus requires frozen exact_opponent_levels=2")

    bundle, behavior, session, partial, stage = staged._new_state(
        seed=int(args.seed),
        domain=args.domain,
        solver=solver,
        args=run_args,
    )
    staged._run_one_iteration(
        seed=int(args.seed),
        domain=args.domain,
        iteration=1,
        bundle=bundle,
        behavior=behavior,
        session=session,
        partial=partial,
        solver=solver,
        args=run_args,
        ensemble_size=int(freeze["ensemble_size"]),
        stage=stage,
    )
    first_checkpoint = dict(stage["checkpoints"][-1])
    if not bool(first_checkpoint["ensemble_frozen_fit_gate_pass"]):
        raise RuntimeError("bootstrap ensemble failed the frozen advantage gate")

    paired_advantage = BottomHashCorpus[PairedSample](int(args.capacity))
    paired_strategy = BottomHashCorpus[PairedSample](int(args.capacity))
    collector = PairedPartialExactCollector(
        policy=behavior,
        terminal_utility=icm_delta_utility(PAYOUT),
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
        paired_advantage=paired_advantage,
        paired_strategy=paired_strategy,
        domain=args.domain,
        corpus_seed=int(args.seed),
    )

    paired_roots = 0
    global_root = int(stage["global_root"])
    scenario_counts = [0] * len(mono._scenario_cycle(args.domain))
    chunk = int(args.initial_paired_roots)
    while True:
        global_root = _collect_chunk(
            solver=solver,
            collector=collector,
            bundle=bundle,
            seed=int(args.seed),
            domain=args.domain,
            iteration=2,
            roots=chunk,
            global_root=global_root,
            scenario_counts=scenario_counts,
            exact_opponent_levels=int(run_args.exact_opponent_levels),
        )
        paired_roots += chunk
        enough = (
            paired_advantage.seen >= int(args.min_advantage)
            and paired_strategy.seen >= int(args.min_strategy)
        )
        if enough or paired_roots >= int(args.max_paired_roots):
            break
        chunk = min(int(args.extension_roots), int(args.max_paired_roots) - paired_roots)
        if chunk <= 0:
            break

    coverage_pass = bool(
        paired_advantage.seen >= int(args.min_advantage)
        and paired_strategy.seen >= int(args.min_strategy)
        and all(count > 0 for count in scenario_counts)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save_samples(args.out_dir / "advantage_pairs.pt", paired_advantage.items)
    _save_samples(args.out_dir / "strategy_pairs.pt", paired_strategy.items)
    report = {
        "schema": SCHEMA,
        "domain": args.domain,
        "corpus_seed": int(args.seed),
        "bootstrap_roots": int(args.bootstrap_roots),
        "bootstrap_checkpoint": first_checkpoint,
        "paired_iteration": 2,
        "paired_roots": int(paired_roots),
        "global_root_start": int(args.bootstrap_roots),
        "global_root_end": int(global_root),
        "scenario_counts_paired_phase": scenario_counts,
        "all_scenarios_exercised_paired_phase": all(count > 0 for count in scenario_counts),
        "advantage": paired_advantage.state_summary(),
        "strategy": paired_strategy.state_summary(),
        "minimum_advantage": int(args.min_advantage),
        "minimum_strategy": int(args.min_strategy),
        "coverage_pass": coverage_pass,
        "candidate_inference_used": false,
        "behavior_observation_wire": "SPNNIV1",
        "paired_secondary_wire": "SPNNIV2",
        "v2_serialization_consumes_rng": false,
        "frozen_behavior_semantic_id": freeze["behavior_semantic_id"],
        "ready_for_tables": false,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if coverage_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
