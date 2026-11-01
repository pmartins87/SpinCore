# SpinCore — Auditoria estática de compatibilidade Deep CFR / SD-CFR para Spin & Go

Data: 2026-11-01  
Escopo: **análise técnica, sem autorização de mudança estratégica**  
Status: **G0 CENTRAL PASS; G1/G2 PASS; G3 EXECUTION PASS / CANDIDATE FAIL; READ-ONLY ROOT CAUSE NEXT**  
Fonte de verdade operacional: `CURRENT_WORK.md`; decisões e gates: `ROADMAP.md`.

## Decisão

**MANTER** a arquitetura e os artefatos atuais como baseline. **NÃO** iniciar o treino longo,
**NÃO** trocar pelo OpenSpiel, **NÃO** ativar SD-CFR, **NÃO** descartar checkpoints ou reservoir.
O benchmark de quatro objetivos permanece preparado, mas **adiado**, sem invalidar seus
contratos, até a auditoria de compatibilidade ser adjudicada.

O propósito é verificar se uma correção baseada em prática publicada, limitada e reversível
pode superar a abordagem atual sem sacrificar o aprendizado acumulado.
Esta revisão **não** demonstra equivalência integral com uma referência, superioridade,
ganho de EV ou disponibilidade física de todos os arquivos no Ryzen/WSL.

## Referências primárias

- Brown, Lerer, Gross, Sandholm (2019), *Deep Counterfactual Regret Minimization*: https://proceedings.mlr.press/v97/brown19b.html
- OpenSpiel PyTorch Deep CFR, referência de implementação: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/deep_cfr.py
- Steinberger (2019), *Single Deep Counterfactual Regret Minimization*: https://arxiv.org/abs/1901.07621
- Steinberger, implementação Deep-CFR/SD-CFR: https://github.com/EricSteinberger/Deep-CFR
- Brown & Sandholm, *Superhuman AI for multiplayer poker* (Pluribus): https://doi.org/10.1126/science.aay2400

OpenSpiel disponibiliza um algoritmo genérico para jogos sequenciais e aceita game.num_players();
**não** é uma implementação validada de Spin & Go 3-max. O trabalho do Pluribus é evidência
de eficácia prática de MCCFR no poker multiplayer, **não** prova de equivalência entre
Pluribus e o SpinCore e nem de convergência a Nash em todo jogo multiplayer.

## Correspondência estática comprovada no código atual

| Item de referência | Evidência no SpinCore | Julgamento |
| --- | --- | --- |
| Percorrer ações do jogador traverser; produzir target Advantage = valor ação - valor médio | `python/spincore/r7_5_action_cfr.py`, `_adv_partial`, `collect_advantage_partial_exact` | **PRESENTE**, com extensão `exact_opponent_levels` |
| Externally sampled actions dos adversários | `r7_5_action_cfr.py`, `_sample_opponent_action` | **PRESENTE**, com ramo alternativo de enumeração exata |
| Reservoir de Advantages e treinamento supervisionado | `r7_5_action_training.py`, `lean_functional_training.py`, `python/spincore_nn/training.py` | **PRESENTE** |
| Reset de Advantage net a cada iteração no caminho lean clássico | `lean_functional_training.py`, `reset_advantage_network` | **PRESENTE** no baseline clássico |
| Rede distinta de estratégia média | `spincore_nn/action_models.py`, `AveragePolicyActionNetV1`; `r7_5_action_training.py` | **PRESENTE** no Deep CFR clássico do projeto |
| Regret matching com Advantages positivos | `python/spincore/lean_action_policy.py`, `lean_regret_matching_policy` | **PRESENTE** |
| Fallback quando nenhuma ação legal tem Advantage positivo | SpinCore: softmax dos raw Advantages; Brown 2019/OpenSpiel PyTorch: argmax determinístico do raw Advantage | **DIVERGÊNCIA CONFIRMADA** |
| Uma Advantage net por jogador (algoritmo de referência) | `lean_functional_training.py`: um `DomainRuntime` por domínio 3H/HU; `collect_root` percorre todos os jogadores vivos | **DIVERGÊNCIA DE PARAMETRIZAÇÃO**: rede compartilhada por domínio; checar que a observação carrega integralmente posição e identidade estratégica |
| Política direta do professor @17315 | `lean_action_policy.py` e scripts `run_3h_semantic_long_15315_17315.py` | **ADAPTAÇÃO PRÓPRIA**: ENS8 final -> média raw -> regret matching. Não equivale por definição a SD-CFR. |
| Utilidade efetiva do treino | `python/spincore/lean_training_scope.py`: `CHIP_EV_WTA_V1`, `chip_delta / 1500`, payout principal `(1,0,0)` | **ESCOLHA EXPLÍCITA**: chipEV para escopo WTA, não ICM de múltiplas premiações |
| Espaço de ações | `python/spincore/lean_action_scope.py`: sete ações ativas e dez slots de rede; sizing preflop contextual, postflop 33/50/75/100/all-in | **ABSTRAÇÃO EXPLÍCITA**: não é NLHE contínuo sem restrições |
| Condições HU/3H | `lean_functional_training.py` e `legacy_scenario.py`: domínios separados e amostragem de blinds/stacks históricos | **PRESENTE**; fidelidade distributiva da amostragem deve ser validada separadamente |

### Importante sobre o fallback

O artigo do Deep CFR observa que, se a soma dos regrets positivos é zero, muitas
estratégias são possíveis; os autores **escolheram o maior regret** para mitigar
erro de aproximação. OpenSpiel PyTorch aplica o mesmo argmax.
A implementação atual do SpinCore utiliza **softmax**, o que pode criar
probabilidades próximas de uniformes entre ações muito distintas.
Isso é uma **diferença concreta de algoritmo**, mas não se autoriza trocar a
regra sem verificar os efeitos sobre estados normais e patológicos.

A auditoria anterior @17315 documentou fallback com targets CFR positivos em
4.676 estados do holdout protegido, e o teste de member-policy-mean fracassou
em jogo direto. Esses fatos justificam priorizar uma verificação **sem novo
treinamento** da diferença publicada de fallback antes de testar novas losses
inventadas, mas não fornecem, sozinhos, prova de melhora de EV.

## Preservação e opções de migração

1. **Preservar exatamente** checkpoint LT2@8100, LT3@10105, resultados HU, arquivos
   `resume_state.pt`, `semantic_long_ensemble_final_17315.pt`, relatórios,
   training/protected split e hashes. Não regravar artefatos originais.
2. O `resume_state.pt` do caminho 3H longo serializa `adv_mem`,
   `semantic_members`, `member_seed_contract`, `sampler_rng_state`,
   `completed_iteration`, além de metadados da linhagem. Isso é evidência de
   que **existe mecanismo de continuidade e reaproveitamento**, não garantia
   de que todos os arquivos estejam presentes/intactos localmente.
3. **Deep CFR de referência:** motor, amostragem e reservoir podem em princípio
   ser mantidos; uma futura refit da mesma estrutura de rede pode reaproveitar
   targets existentes. Paridade completa não está demonstrada, particularmente
   em rede por jogador versus rede compartilhada e estratégia média.
4. **SD-CFR:** requer políticas/Advantage networks em múltiplas iterações e
   seleção da rede por iteração com ponderação apropriada ao longo da trajetória.
   ENS8 de **oito membros da iteração final** não substitui automaticamente
   essa coleção temporal. Inventariar checkpoints intermediários antes de
   afirmar se um SD-CFR fiel pode ser executado sem novos ajustes.
5. Mudar dimensionalidade, heads ou representação impede herdar pesos antigos
   diretamente sem adaptação e testes; **reservoir/targets continuam potencialmente
   úteis** para refit offline, sujeito a conferir semântica/escopo.
6. Como a política existente foi explicitamente treinada sob `CHIP_EV_WTA_V1`,
   mudar payouts/objetivo de recompensa não é mera troca de interface.
   O solver expõe `terminal_icm_delta`, mas isso **não** demonstra que o
   treinamento atual usa ICM multi-place.

## Gates enxutos antes de qualquer recomendação de mudança

- **G0 — inventário read-only (presença/tamanho confirmado; SHA256/metadados pendentes):** existência, tamanho, SHA256
  e metadados de: checkpoint @10105; HU sidecar; @15315/@17315 ensembles;
  @17315 resume; relatórios PASS; split protegido. Confirmar que snapshots
  históricos SD-CFR realmente existem antes de contar com eles.
- **G1 — conformidade executável em jogo pequeno (não executado):**
  teste de referência de regret matching, máscaras legais, amostragem traverser,
  target de regret, pesos de iteração e reset; comparar semântica, não exigir
  igualdade bitwise de arquiteturas intencionalmente distintas.
- **G2 — diferença de fallback, apenas inferência (não executado):**
  em estados congelados, comparar atual SOFTMAX_FALLBACK versus fallback
  ARGMAX publicado, **sem** treinar/alterar a política atual; verificar
  target-fidelity, frequência de all-ins absurdos, folds de monstros e
  conservação de ações legais. Se não houver evidência consistente, não promover.
- **G3 — somente se G2 indicar candidato:** teste de jogo direto com amostra
  pré-registrada e regras de parada; não considerar redução de um único flag
  como prova suficiente de maior EV.
- **Critério de migração:** só propor alteração de arquitetura com
  (i) compatibilidade com 3H/HU, (ii) baixo desperdício de dados existentes,
  (iii) demonstração de vantagem material e (iv) rollback reprodutível.

**Não** executar o benchmark de losses pendente como reação reflexa ao
diagnóstico; ele permanece preparado e poderá voltar a ser prioritário se
a inspeção baseada em algoritmo publicado não explicar o problema.

## Limitações explícitas

- Revisão do código do branch principal e de documentação existente; **sem**
  executar teste local, consultar filesystem remoto do Ryzen ou validar em mesa.
- A implementação OpenSpiel não cobre por si só as regras específicas de Spin & Go.
- Evidência de sucesso de métodos CFR em poker multiplayer não implica a mesma
  garantia de convergência conhecida para dois jogadores zero-soma.
- Os checkpoints `*.pt` e reservoirs são locais e não estão no GitHub;
  a presença física e os tamanhos foram confirmados por listagem do usuário;
  integridade SHA256/conteúdo ainda precisa ser certificada no Ryzen/WSL.
- O benchmark de quatro losses corrigido e seu contrato continuam
  **PENDENTES**, não PASS.

## G0 — presença de artefatos confirmada no Ryzen/WSL (relato de 2026-11-01)

Comando executado pelo usuário (somente leitura):
```powershell
wsl -e bash -lc "cd ~/spincore_lean_functional && ls -lh runs/3h_semantic_long_15315_17315/ runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt"
```

Arquivos encontrados:
- `runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt` — **2.7G** (Sep 23 11:00).
- `runs/3h_semantic_long_15315_17315/resume_state.pt` — **632M** (Oct 25 02:36).
- `runs/3h_semantic_long_15315_17315/semantic_long_ensemble_15815.pt` — **5.0M** (Oct 20 04:57).
- `runs/3h_semantic_long_15315_17315/semantic_long_ensemble_16315.pt` — **5.0M** (Oct 21 20:21).
- `runs/3h_semantic_long_15315_17315/semantic_long_ensemble_16815.pt` — **5.0M** (Oct 23 11:31).
- `runs/3h_semantic_long_15315_17315/semantic_long_ensemble_17315.pt` — **5.0M** (Oct 25 02:35).
- `runs/3h_semantic_long_15315_17315/semantic_long_ensemble_final_17315.pt` — **5.0M** (Oct 25 02:36).
- `3h_semantic_long_15315_17315.json` — **5.1M**; `memory_parallel.log` — **2.9M**; `training_parallel.log` — **5.5M**.
- `parallel_semantic_pack/` directory exists; content not enumerated.

**G0 finding:** multiple *time-separated* Advantage-ensemble snapshots exist;
this strengthens the case for **reusing** previous work. However, these
are approximately 500-iteration snapshots, **not proof of a complete per-iteration
SD-CFR archive**. Earlier runs, HU artifacts, protected split, and their hashes
were not included in this narrow listing. Checkpoint presence does not yet prove
loadability or correct hash binding to published PASS reports.

Next minimal command (no Bash shell, no new Python package, no mutation):
```powershell
wsl -e sha256sum /home/rz9/spincore_lean_functional/runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt /home/rz9/spincore_lean_functional/runs/3h_semantic_long_15315_17315/resume_state.pt /home/rz9/spincore_lean_functional/runs/3h_semantic_long_15315_17315/semantic_long_ensemble_final_17315.pt
```

Cross-check against pinned source SHA256 `f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0`
and teacher SHA256 `d856acacf4f9d1e03f424c15a1d41cc36bb0384f3cbc62e6fadea84979c6e9f4` listed
in the @17315 experiment manifest. The resume hash must be confirmed using its
associated original PASS report; **do not infer** it from file size.

### Utility nuance

For a **winner-take-all** payout vector under the *Independent Chip Model*,
the modeled probability of first place equals chip fraction `stack/total_chips`.
Thus chipEV optimization is consistent with the WTA *ICM proxy* up to constant
scale; this is not proof that chip fraction equals real-world first-place equity
under strategic tournament dynamics. Multi-place payout vectors remain distinct.

### Reference nuance

OpenSpiel PyTorch Deep CFR implements argmax(raw legal Advantage) on an
all-nonpositive neural output, while its generic tabular CFR fallback may be
uniform. The comparison must use the **Deep CFR neural reference** and avoid
conflating the two algorithms.

## G0 — SHA256 de arquivos principais (verificação do usuário)

O usuário executou `wsl -e sha256sum` diretamente (somente leitura):

| Arquivo | SHA256 observado | Referência | Resultado |
| --- | --- | --- | --- |
| `runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt` | `f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0` | Checkpoint imutável @10105 em `ROADMAP.md`, `pilot.EXPECTED_SHA` | **PASS SHA** |
| `runs/3h_semantic_long_15315_17315/semantic_long_ensemble_final_17315.pt` | `d856acacf4f9d1e03f424c15a1d41cc36bb0384f3cbc62e6fadea84979c6e9f4` | `contracts/run_manifests/semantic_objective_benchmark_17315.json` e `CURRENT_WORK.md` | **PASS SHA** |
| `runs/3h_semantic_long_15315_17315/resume_state.pt` | `ed0e00a05fe3c6900100b12de5f27efcaf24261adb816090320cc9b34b1d403f` | Chave `resume_state_sha256` do relatório local, enviada pelo usuário | **PASS SHA** |

A própria implementação original `tools/run_3h_semantic_long_15315_17315.py`
preenche `resume_state_sha256` no relatório antes de imprimi-lo ao concluir
(`SEMANTIC_LONG_15315_17315_TRAINING_PASS`). Logo, basta ler a chave no relatório
local, sem carregar checkpoint, para conferir a integridade do resume.

Comando somente leitura sem intermediário Bash (PowerShell):
```powershell
wsl -e grep resume_state_sha256 /home/rz9/spincore_lean_functional/runs/3h_semantic_long_15315_17315/3h_semantic_long_15315_17315.json
```

Se o valor for igual a `ed0e00a05fe3c6900100b12de5f27efcaf24261adb816090320cc9b34b1d403f`,
marcar **PASS SHA** para o resume. Caso divirja, não assumir corrupção: investigar
por que o relatório e o arquivo diferem, antes de qualquer ação.

**Não** atestar integridade dos snapshots intermediários, HU ou reservoirs
não verificados. Nenhuma mudança de inferência ou treino está autorizada.

## G0 — verificação do resume concluída (2026-11-01)

O usuário executou `wsl -e grep resume_state_sha256 .../3h_semantic_long_15315_17315.json` e retornou
`"resume_state_sha256": "ed0e00a05fe3c6900100b12de5f27efcaf24261adb816090320cc9b34b1d403f"`.
Coincide **exatamente** com `sha256sum` calculado para `resume_state.pt`.

**Resultado: G0-CENTRAL-3-SHA PASS**: checkpoint @10105, final ENS8 @17315 e resume @17315.
Isto comprova igualdade criptográfica com os registros de referência verificados,
mas **não** comprova cobertura integral do histórico SD-CFR, validade da estratégia,
passagem em teste de load, nem hashes dos snapshots adicionais.

## G1/G2 — diagnóstico offline preparado (não executado no Ryzen)

Foi criado `tools/audit_reference_fallback_17315.py` com **somente leitura**,
sem novos CFR roots, re-treino, mutação do professor ou substituição da política.

- `--self-test` compara a implementação real do SpinCore à forma de
  regret matching publicada para estados sintéticos de regret positivo,
  all-nonpositive, empate e restrição de ações legais.
  Esta etapa valida **apenas a função de política**, não toda a travessia Deep CFR.
- `--raw-report` reutiliza o relatório local já criado pelo diagnóstico
  `run_3h_teacher_raw_advantage_17315.sh` em
  `runs/teacher_raw_advantage_17315/teacher_raw_advantage_17315.json`.
  Reconstitui as políticas atuais a partir do vetor raw, exige paridade com
  as probabilidades armazenadas e calcula a política de fallback
  `ARGMAX_LEGAL_RAW` como contrafactual estritamente por estado.
- Informa TV das distribuições e probabilidade média de ALL_IN entre estados
  congelados, por grupo sinalizado. Não reexecuta mãos; não mede EV.
  **A seleção original de estados sinalizados é enviesada**; conclusões
  não podem ser extrapoladas ao conjunto completo de decisões.
- Em falhas de paridade ou schema, falha explicitamente em vez de produzir
  recomendações. Nenhum efeito na estratégia de produção é autorizado.

Comando recomendado no PowerShell (sem quoting aninhado com Bash vars):

```powershell
wsl -e bash -lc "cd ~/spincore_lean_functional && git pull --ff-only origin main && .venv_lean/bin/python tools/audit_reference_fallback_17315.py --self-test --raw-report runs/teacher_raw_advantage_17315/teacher_raw_advantage_17315.json"
```

**Gates seguintes:** após saída G1/G2, adjudicar relevância; se houver
sinal de melhoria, pré-registrar jogo direto sobre cenários frescos. Um
resultado positivo em snapshots enviesados **não** é autorização de
substituição do fallback. Preservar todos os checkpoints e o benchmark de
quatro objetivos, que continua pendente/adiado.

## G1/G2 — executed on Ryzen: PASS (2026-11-01)

The user executed the published-fallback diagnostic after pulling the source.
Observed:
- `G1_REFERENCE_FALLBACK_SEMANTICS_SELFTEST_PASS`;
- `SPINCORE_REFERENCE_FALLBACK_FROZEN_DIAGNOSTIC_17315_PASS`;
- schema `SPINCORE_REFERENCE_FALLBACK_FROZEN_DIAGNOSTIC_17315_V1`;
- `baseline_policy_parity=true`; **367** frozen flagged/extreme decision rows;
- **169/367** used the all-nonpositive fallback and changed under published
  argmax; total mean TV over selected rows **0.3505833**.
- 18 sampled deep-72o all-ins: current mean all-in **0.3318054**, reference
  **0.0**. 20 high-probability deep-72o opportunities: **0.3326141 -> 0**;
- 271 sampled high-card jams: overall mean all-in **0.4410964 -> 0.3737804**.
  Within 129 fallback decisions with legal all-in, **0.2034312 -> 0.0620155**;
- The 20 extreme *high-card-no-draw* opportunity rows had **zero** fallback
  events and **no change** in mean all-in: **0.9695501 -> 0.9695501**.
  The distinct top-20 high-card jam list also remains at all-in **1.0**.
- Trips+/monster/AA opportunity rows cannot be pronounced fixed: among
  sampled trips+ folds, three of four used fallback, but only all-in
  probabilities were summarized in G1/G2, not fold correctness.

**Adjudication:** published argmax plausibly addresses the *fallback-driven*
72o pathology, not the separate *positive-regret high-card/no-draw*
amplification/model-sign failure. The 367 rows are baseline-selected
and biased; **no EV, deployment, or global safety conclusion** follows.

## G3 — bounded common-chance direct-play gate prepared, NOT EXECUTED

The only candidate is the **published Deep CFR fallback choice**
`argmax(raw legal Advantage)` when every legal Advantage is nonpositive.
Everything else remains strictly the same: original @17315 eight-member
semantic Advantage weights, raw-member arithmetic mean, positive-Advantage
regret matching, same 3H scenarios/opponent/engine, 10k fresh fixed-seed
paired common-chance scenarios, 16 workers. No new CFR roots or optimizer steps.

Preregistered **before** direct-play results:
- `contracts/reference_fallback_direct_play_17315_v1.json` (frozen seed
  `2026110103`, 10000 scenarios, criteria including 95% paired EV
  noninferiority lower bound >= -0.50 chips/seat-hand, 72o improvement,
  no new critical folds, high-card-no-draw and trips-fold noninferiority);
- `tools/validate_reference_fallback_direct_play_17315.py`, reuses the
  *existing* SpinCore common-chance engine, scenario summary and sanity
  metrics and the **original** semantic-vector implementation. Includes
  selftest and a 4-scenario smoke pass prerequisite;
- `tools/run_reference_fallback_direct_play_17315.sh` uses read-only source
  hashes via `contracts/run_manifests/reference_fallback_direct_play_17315.json`,
  checks working tree/RAM/heavy trainer, performs selftest/smoke before the
  full run and attempts to copy the report to Desktop.

A PASS identifies an *inference-only candidate* for a later integration
gate. A FAIL rejects it as a direct replacement. Neither outcome permits
automatic production changes or long 3H/HU training. This experiment cannot
resolve the positive-regret high-card/no-draw pathology.

PowerShell:
```powershell
wsl -e bash -lc "cd ~/spincore_lean_functional && git pull --ff-only origin main && bash tools/run_reference_fallback_direct_play_17315.sh"
```

## G3 — executed on Ryzen (2026-11-01): technical PASS, strategic FAIL

All prerequisites and execution gates passed:
`STAGE_CONTRACT_PASS`, `G1_REFERENCE_FALLBACK_SEMANTICS_SELFTEST_PASS`,
`G3_FALLBACK_ONLY_SELFTEST_PASS`, smoke `PASS`, 10,000 each A/B fresh
paired 3H scenarios, report `PASS`, postvalidation `PASS`, runner `PASS`.
Wall clock: **8m30.77s**. Actual result **`G3_DECISION=FAIL`**.

Comparison A=current softmax fallback; B=the exact same frozen teacher,
but argmax fallback only on all-nonpositive legal raw Advantages.

| Metric | A | B | Interpretation |
| --- | ---: | ---: | --- |
| Deep-72o mean ALL_IN | 0.0132795299 | 0.0036111909 | Difference 0.00966834 < required 0.01; **FAIL** (sole failing criterion) |
| Deep-72o opportunities | 584 | 585 | count gate PASS |
| Deep-72o sampled jam flags | 8 | 2 | fewer, but count not sufficient to override gate |
| High-card/no-draw mean ALL_IN | 0.03879102 | 0.01302104 | improved, check PASS |
| High-card/no-draw fraction >=80% ALL_IN | 0.00744048 | 0.00655165 | noninferiority PASS, extreme tail remains |
| High-card jam flags | 230 | 134 | fewer, but no proof all positive-regret failures resolved |
| Trips+ mean fold | 0.03990495 | 0.05631636 | worsened 1.641pp; within frozen +2pp tolerance |
| Trips+ sampled fold flags | 3 | 9 | **safety warning** despite numeric gate passing |

Paired B-A chipEV/policy-seat-hand: mean **+0.2060667**, standard error
**0.2859632**, 95% CI **[-0.3544213,+0.7665546]**. Meets frozen
noninferiority threshold lower bound >= -0.50, but **does not demonstrate
superiority**.

The G3 deep-72o improvement requirement was **at least 1.00 percentage
point absolute**. Observed reduction was **0.96683 percentage point**,
short by **0.03317 percentage point**. Do not adjust the threshold after
results; the attempt is **rejected as direct replacement**.

The lower global high-card-no-draw mean is from *fresh policy trajectories*,
where states visited differ between A and B. It does **not** contradict
G1/G2's observation that the *frozen selected positive-regret top20*
high-card/no-draw states are completely unaffected by fallback choice.

Observed console provenance preserved in
`contracts/reference_fallback_direct_play_17315_v1_console_result.json`.
Original full report on Ryzen/Desktop is **not attached here** and has not
been re-read; do not claim full independent audit of that artifact.

### Decision and next step

- Keep baseline A; do not promote B, rerun for a favorable seed, or relax criteria.
- Preserve original @10105 checkpoint, @17315 ensemble/resume and milestone
  snapshots; no new CFR roots and no long 3H/HU training.
- Next useful work is **read-only** source/evidence audit of extreme
  **positive-regret** all-in behavior in high-card/no-draw states and of the
  published per-player Advantage vs SpinCore shared-domain network mapping.
  Do not invent a new loss or train a network unless evidence warrants it.
- The four-loss benchmark remains READY but **deferred**, not invalidated.

## G3 — revisão independente do relatório completo anexado (2026-11-01)

O JSON ORIGINAL foi lido: SHA256 `fd91b7a871108d095a1b4fe3d152f4f68a0b8a9557bd4be5932294123a3e70f8`,
151.681 bytes, schema/status/seed/contagens/gates compatíveis com console.
O arquivo completo permanece anexo à conversa, não foi copiado ao GitHub.
Resumo derivado e verificado: `contracts/reference_fallback_direct_play_17315_v1_console_result.json`.

### EV e interpretação dos flags
Cada braço teve 10.000 cenários, 60.000 jogos balanceados; 103.807
decisões SpinCore em A e 106.292 em B. B-A EV pareado +0,20607,
IC95% [-0,35442,+0,76655]: NÃO demonstra superioridade; ambos
obtiveram resultado negativo contra DeepCrusher de desenvolvimento
(-5,37451 A, -4,96238 B fichas no comparador), sujeito às limitações
do benchmark. G3 permanece **FAIL** conforme critério pre-registrado de
melhoria de 72o: 0,96683 pp < 1,00000 pp. Não relaxar limiar.

High-card sem draw: 3.360 vs 4.579 oportunidades; p(ALL_IN) média
3,8791% -> 1,3021%, porém cauda >=95% não melhorou:
19/3360 (0,5655%) -> 27/4579 (0,5896%). Entre as listas de top20,
12 snapshots idênticos (mesmos scenario/lineup/decision/cartas/pot/call)
mantiveram p(ALL_IN)=1.0 nas duas políticas. Como as trajetórias são
diferentes, comparações de médias/contagens não são efeito por estado;
ausência nas listas top20 não implica ausência na trajetória.

### Auditoria dos sete B-folds p=1 com categoria trips+
O flag `POSTFLOP_TRIPS_PLUS_FOLD` apenas marca categoria>=3 pela
`decision_sanity.py`, independentemente de ser mão privada forte.
O resumo completo mostra sete top-probability folds determinísticos:

| Cenário | Hole | Board | Interpretação | to_call/pot |
| --- | --- | --- | --- | --- |
| 367 | 6d3h | Ts Jh Js Jd | JJJ todo no board, kicker baixo | 40/80 |
| 3866 | 2c5h | Qc 8c Qh Qd | QQQ todo no board, kicker baixo | 562/622 |
| 4384 | 2cTd | 7d Jd 8d 7s Qd | Flush T-high, board pareado | 531/591 |
| 5920 | 9c4c | 8d 8h 8s 3s | 888 todo no board | 528/568 |
| 6213 | 3s2s | Jc 8s Ad Js Qs | Flush Q-high com cartas baixas, board pareado | 393/588 |
| 6510 | 8h2h | 7s 9s Ts Td Jc | Sequência J-high, board pareado | 641/681 |
| 9790 | Th8d | Js 7s 7c 7h Qh | 777 todo no board | 402/582 |

Quatro trincas são inteiramente de mesa, dois flushes baixos relativos
ao board pareado e uma sequência de board pareado. To-call/pot 50%-94%.
São **alertas heurísticos**, NÃO sete erros demonstrados. No caso 4384,
o mesmo estado aparece no top20 dos dois braços: A p(FOLD)=0,3352283,
B p(FOLD)=1,0, evidenciando mudança de política por fallback, mas não
a incorreção do fold. No agregado, 3/109 (2,75%) flags em A vs 9/159
(5,66%) em B, com denominadores e trajetórias diferentes; não declarar
degradação estratégica comprovada. Por outro lado, não chamar esses
folds de corretos sem análise de ranges/equity.

Os dois 72o all-in snapshots de B com p(ALL_IN)=1.0 tampouco podem ser
atribuídos ao ramo positivo/negativo apenas com o G3: o relatório não
contém os vetores de Advantage nessas decisões.

### Decisão
Manter **G3 FAIL**, baseline intacto, treinamento bloqueado, nenhum novo
teste com outras sementes para caçar PASS. A nova evidência qualifica e
retira a anterior inferência de que 3->9 flags prova degradação.
A investigação seguinte deve usar os **vetores crus já existentes**
em `runs/teacher_raw_advantage_17315/teacher_raw_advantage_17315.json`
para identificar a origem de ALL_IN extremos com regret positivo,
sem novos CFR roots nem retraining.
