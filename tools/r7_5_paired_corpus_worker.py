from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import r7_4_stability_pilot_worker as mono
import r7_4_staged_domain_worker as staged
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from spincore.deep_cfr import icm_delta_utility, sample_action
from spincore.r7_5_paired_corpus import BottomHashCorpus, PairedSample
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import AdvantageSample, StrategySample

PAYOUT = (0.5, 0.3, 0.2)
SCHEMA = "SPINCORE_R7_5_PAIRED_CORPUS_V1"


class _DiscardMemory:
    """Memory sink used during the frozen paired-collection phase.

    The accepted behavior ensemble is already fitted before paired collection.
    Mutating the old R7.4 reservoirs cannot affect that frozen policy, so keeping
    duplicate V1 samples would only waste memory and advance reservoir-local RNG.
    Traversal RNG and targets remain exactly those of the accepted collector.
    """

    def add(self, _sample) -> None:
        return None


class PairedPartialExactCollector(PartialExactAdvantageCollector):
    """Accepted R7.4 collector plus side-effect-free SPNNIV2 pairing.

    The recursion, behavior-policy calls, opponent exact-enumeration depth,
    sampled branches, weights and return accounting intentionally mirror
    `PartialExactAdvantageCollector` / `ExternalSamplingCollector` exactly.
    The only extra operation is serializing SPNNIV2 at the same sample state and
    retaining the immutable pair in BottomHashCorpus, which consumes no RNG.
    """

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
    def _legal_mask(legal: tuple[int, ...]) -> tuple[int, ...]:
        legal_set = set(int(action) for action in legal)
        return tuple(1 if action in legal_set else 0 for action in range(6))

    def _add_pair(
        self,
        *,
        kind: str,
        state,
        observation_v1: bytes,
        legal: tuple[int, ...],
        target,
        weight: float,
        iteration: int,
    ) -> None:
        sample = PairedSample(
            kind=kind,
            domain=self.domain_name,
            corpus_seed=self.corpus_seed,
            observation_v1=observation_v1,
            observation_v2=state.neural_bytes_v2(),
            legal=self._legal_mask(legal),
            target=tuple(float(value) for value in target),
            weight=float(weight),
            iteration=int(iteration),
        )
        (self.paired_advantage if kind == "advantage" else self.paired_strategy).add(sample)

    def _adv_partial(
        self,
        state,
        traverser: int,
        iteration: int,
        exact_levels: int,
        explicit_opponent_mass: float,
    ):
        if state.terminal:
            return float(self.terminal_utility(state)[traverser]), 1, 0

        actor = state.actor
        legal = state.legal_actions()
        observation = state.neural_bytes()
        sigma = self._p(state, observation, legal)

        if actor == traverser:
            values = [0.0] * 6
            nodes = 1
            added = 0
            for action in legal:
                child = state.child(action)
                try:
                    value, child_nodes, child_added = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_levels,
                        explicit_opponent_mass,
                    )
                finally:
                    child.close()
                values[action] = float(value)
                nodes += int(child_nodes)
                added += int(child_added)

            node_value = sum(float(sigma[action]) * float(values[action]) for action in legal)
            target = [0.0] * 6
            for action in legal:
                target[action] = float(values[action]) - float(node_value)

            if explicit_opponent_mass > 0.0:
                weight = float(iteration) * float(explicit_opponent_mass)
                legal_mask = self._legal_mask(legal)
                self.advantage_memory.add(
                    AdvantageSample(
                        observation,
                        legal_mask,
                        tuple(target),
                        weight,
                        int(iteration),
                    )
                )
                self._add_pair(
                    kind="advantage",
                    state=state,
                    observation_v1=observation,
                    legal=legal,
                    target=target,
                    weight=weight,
                    iteration=iteration,
                )
                added += 1
            return float(node_value), int(nodes), int(added)

        if exact_levels > 0:
            value = 0.0
            nodes = 1
            added = 0
            for action in legal:
                probability = float(sigma[action])
                if probability <= 0.0:
                    continue
                child = state.child(action)
                try:
                    child_value, child_nodes, child_added = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_levels - 1,
                        float(explicit_opponent_mass) * probability,
                    )
                finally:
                    child.close()
                value += probability * float(child_value)
                nodes += int(child_nodes)
                added += int(child_added)
            return float(value), int(nodes), int(added)

        action = sample_action(sigma, legal, self.rng)
        child = state.child(action)
        try:
            value, nodes, added = self._adv_partial(
                child,
                traverser,
                iteration,
                0,
                explicit_opponent_mass,
            )
        finally:
            child.close()
        return float(value), int(nodes) + 1, int(added)

    def _strategy(self, state, target_player: int, iteration: int):
        if state.terminal:
            return 0

        actor = state.actor
        legal = state.legal_actions()
        observation = state.neural_bytes()
        sigma = self._p(state, observation, legal)

        if actor == target_player:
            weight = float(iteration)
            legal_mask = self._legal_mask(legal)
            self.strategy_memory.add(
                StrategySample(
                    observation,
                    legal_mask,
                    tuple(sigma),
                    weight,
                    int(iteration),
                )
            )
            self._add_pair(
                kind="strategy",
                state=state,
                observation_v1=observation,
                legal=legal,
                target=sigma,
                weight=weight,
                iteration=iteration,
            )
            action = sample_action(sigma, legal, self.rng)
            child = state.child(action)
            try:
                return 1 + self._strategy(child, target_player, iteration)
            finally:
                child.close()

        total = 0
        for action in legal:
            child = state.child(action)
            try:
                total += self._strategy(child, target_player, iteration)
            finally:
                child.close()
        return total


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
    seed: int,
    domain: str,
    iteration: int,
    roots: int,
    global_root: int,
    scenario_counts: list[int],
    exact_opponent_levels: int,
) -> tuple[int, dict[str, int]]:
    scenarios = mono._scenario_cycle(domain)
    live_by_scenario = [tuple(i for i, stack in enumerate(ep.stacks) if stack > 0) for ep in scenarios]
    stats = {"roots": 0, "nodes": 0, "advantage_samples": 0, "strategy_samples": 0}

    for _ in range(int(roots)):
        scenario_index = global_root % len(scenarios)
        episode = scenarios[scenario_index]
        live = live_by_scenario[scenario_index]
        scenario_counts[scenario_index] += 1
        deck_seed = (int(seed) * 1_000_003 + global_root * 97 + int(iteration)) & ((1 << 64) - 1)

        for traverser in live:
            root = solver.create(episode, deck_seed)
            try:
                result = collector.collect_advantage_partial_exact(
                    root,
                    traverser=int(traverser),
                    iteration=int(iteration),
                    exact_opponent_levels=int(exact_opponent_levels),
                )
            finally:
                root.close()
            stats["nodes"] += int(result.nodes)
            stats["advantage_samples"] += int(result.samples_added)

        for target_player in live:
            root = solver.create(episode, deck_seed)
            try:
                stats["strategy_samples"] += int(
                    collector.collect_strategy_own_reach(
                        root,
                        target_player=int(target_player),
                        iteration=int(iteration),
                    )
                )
            finally:
                root.close()

        global_root += 1
        stats["roots"] += 1

    return global_root, stats


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

    if int(args.bootstrap_roots) <= 0 or int(args.initial_paired_roots) <= 0:
        raise ValueError("root budgets must be positive")
    if int(args.extension_roots) <= 0 or int(args.max_paired_roots) < int(args.initial_paired_roots):
        raise ValueError("invalid paired coverage-extension budget")

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
        advantage_memory=_DiscardMemory(),
        strategy_memory=_DiscardMemory(),
        paired_advantage=paired_advantage,
        paired_strategy=paired_strategy,
        domain=args.domain,
        corpus_seed=int(args.seed),
    )

    paired_roots = 0
    global_root = int(stage["global_root"])
    global_root_start = global_root
    scenario_counts = [0] * len(mono._scenario_cycle(args.domain))
    totals = {"roots": 0, "nodes": 0, "advantage_samples": 0, "strategy_samples": 0}
    chunk = int(args.initial_paired_roots)

    while True:
        global_root, chunk_stats = _collect_chunk(
            solver=solver,
            collector=collector,
            seed=int(args.seed),
            domain=args.domain,
            iteration=2,
            roots=chunk,
            global_root=global_root,
            scenario_counts=scenario_counts,
            exact_opponent_levels=int(run_args.exact_opponent_levels),
        )
        paired_roots += chunk
        for key in totals:
            totals[key] += int(chunk_stats[key])

        enough = (
            len(paired_advantage.items) >= int(args.min_advantage)
            and len(paired_strategy.items) >= int(args.min_strategy)
        )
        if enough or paired_roots >= int(args.max_paired_roots):
            break
        chunk = min(int(args.extension_roots), int(args.max_paired_roots) - paired_roots)
        if chunk <= 0:
            break

    coverage_pass = bool(
        len(paired_advantage.items) >= int(args.min_advantage)
        and len(paired_strategy.items) >= int(args.min_strategy)
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
        "global_root_start": int(global_root_start),
        "global_root_end": int(global_root),
        "scenario_counts_paired_phase": scenario_counts,
        "all_scenarios_exercised_paired_phase": all(count > 0 for count in scenario_counts),
        "paired_phase_traversal_counts": totals,
        "advantage": paired_advantage.state_summary(),
        "strategy": paired_strategy.state_summary(),
        "minimum_advantage": int(args.min_advantage),
        "minimum_strategy": int(args.min_strategy),
        "coverage_pass": coverage_pass,
        "candidate_inference_used": False,
        "behavior_observation_wire": "SPNNIV1",
        "paired_secondary_wire": "SPNNIV2",
        "v2_serialization_consumes_rng": False,
        "frozen_behavior_semantic_id": freeze["behavior_semantic_id"],
        "ready_for_tables": False,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if coverage_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
