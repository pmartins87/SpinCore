# SpinCore — Auditoria estática de compatibilidade Deep CFR / SD-CFR para Spin & Go

Data: 2026-11-01  
Escopo: **análise técnica, sem autorização de mudança estratégica**  
Status: **G0 TRÊS ARTEFATOS CENTRAIS PASS SHA256; DEMAIS ARTEFATOS E G1/G2 PENDENTES**  
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
