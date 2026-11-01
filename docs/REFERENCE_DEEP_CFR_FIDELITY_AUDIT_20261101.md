# SpinCore — Auditoria finita de fidelidade estrutural ao Deep CFR (2026-11-01)

**Estado: CODE-READ AUDIT COMPLETE / NO CHANGE APPROVED / REACHABILITY AND IMPACT GATES NOT EXECUTED.**
Esta auditoria **não** executa treinamento, não altera modelo, não usa sealed holdout,
não substitui arquitetura e não afirma que Deep CFR garante convergência
de Nash em Spin & Go three-handed.

## Governança consultada

Lidos `AGENTS.md`, `PROJECT_CONTRACT.yaml`, `contracts/AUDIT_STATUS.yaml`,
`contracts/action_and_representation.yaml`, `contracts/training_and_models.yaml`,
`contracts/source_of_truth.yaml`, `contracts/validation_and_benchmarks.yaml`,
`docs/LEGACY_BASELINE_AND_QUALITY_POLICY.md`,
`docs/RYZEN_OPTIMIZATION_POLICY.md`, `ROADMAP.md` e `CURRENT_WORK.md`.
IDs principais em análise: **REP-001, REP-002, REP-010, REP-011, REP-012,
REP-013, REP-015, REP-018, ACT-010, TRAIN-001, TRAIN-002, TRAIN-003,
TRAIN-004, TRAIN-005, TRAIN-006, TRAIN-007, MODEL-001, MODEL-020,
MODEL-025, VALID-002, SRC-001, SRC-002, SRC-004**.

Fontes publicadas:
- Brown et al. (2019), seção 4 e algoritmos 1–2 (p. 4–6):
  https://proceedings.mlr.press/v97/brown19b/brown19b.pdf
- Código de referência OpenSpiel PyTorch:
  https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/deep_cfr.py

## Resultado por elemento

### 1. Calculadora de targets de Advantage — sem bug estrutural encontrado

`python/spincore/r7_5_action_cfr.py`, linhas 250–344:
- nó do traverser: avalia todos os slots legais efetivos;
- `node_value = sum(sigma[a] * value[a])`;
- `target[a] = value[a] - node_value`;
- nós dos demais jogadores: sorteiam ação segundo `sigma`,
  salvo ramificação explicitamente parcial-exata;
- pesos da amostra: `iteration * explicit_opponent_mass`.
`python/spincore/lean_solver_actions.py` chama o resolver nativo
para ações legais e filhos, preservando a deduplicação de ações exatas.
`python/spincore/r7_5_action_training.py` percorre cada jogador vivo
como traverser e usa o mesmo deck seed dentro de cada root.

**Juízo: estruturalmente alinhado com external-sampling CFR;**
não atesta ausência de bugs numéricos/semânticos em todos os estados
e não certifica targets individuais contra solução exata.

### 2. Identidade, posição e informação privada — presente em coordenadas relativas

`src/hand_infoset_adapter.cpp` cria `CanonicalInfoset` do ator corrente:
somente as duas hole cards do ator e o bordo público visível,
pilhas/compromissos/status em ordem `(actor+k)%3` e
`dealer_rel` relativo ao ator.
`src/neural_encoder.cpp` preserva `dealer_rel`, `domain`,
`street`, `live_count` e as pilhas/compromissos relativos;
`python/spincore_nn/action_models.py` codifica esses campos.

**Juízo: não há evidência de bug por falta de um campo de ID absoluto
do jogador.** Em self-play com assentos simétricos a representação
relativa é intencional e potencialmente adequada. Isso não prova
identificabilidade de todas as histórias.

### 3. Histórico de ações V1 — perda concreta de campos; impacto ainda não provado

`src/hand_infoset_adapter.cpp` preserva `public_events` com
`actor_rel`, tipo, valor pago, compromisso resultante, pote anterior/
posterior e indicador `forced`. Mas a sequência `public_history`
para V1 é codificada somente como
`tok = 1 + action.type + 8 * street`.
`src/neural_encoder.cpp` copia os **últimos 32** tokens para `NeuralInputV1`.

Logo V1 **não preserva explicitamente quem tomou cada ação nem o
respectivo valor histórico** na sequência da rede; a geometria atual
de stacks/commitments/pot contém apenas um resumo parcial.
`src/neural_encoder_v2.cpp`, linhas 248–262, **já** contém
eventos mais ricos com `actor_rel`, `paid`,
`resulting_commitment`, etc. A família semântica @17315
`python/spincore_nn/semantic_v1.py` acrescenta apenas features
derivadas das cartas privadas e bordo, portanto **não reconstitui**
o histórico que V1 perdeu.

**Juízo: limitação de fidelidade da observação V1 demonstrada pela fonte;
não foi demonstrada colisão estrategicamente decisiva em estados legais
nem relação causal com os all-ins @17315.**
`REP-011/REP-012` exigem contexto estratégico de trajetória; já
`REP-015/REP-018` mantêm V1 como fallback estável apesar do
histórico grosso. Essa tensão exige evidência e adjudicação,
não uma promoção silenciosa para V2/V3. Migração quebraria a
compatibilidade direta de pesos e requer suas próprias garantias.

### 4. Máscaras e ações legais — pipeline coerente, sem deriva visível

O V1 nativo contém máscara histórica de **6 slots**, mas o
modelo universal possui **10 slots**; o treino de R7.5
e o caminho lean geram máscaras legais de 10 slots
separadamente com `r7_5_action_cfr.legal_mask(legal)`.
O output de 10 ações é mascarado na loss pelo vetor efetivo legal;
`lean_action_policy.py` calcula p apenas sobre `legal`.
`lean_action_scope.py` preserva as sete primitivas ativas,
deduplicadas pelo solver nativo.

**Juízo: nenhuma incompatibilidade de índice de ação demonstrada
neste pipeline.** Não presumir que a máscara histórica de seis
slots é usada diretamente como máscara de loss universal.

### 5. Objetivo de regressão — divergência matemática comprovada

Deep CFR, Brown et al. (2019), algoritmo 1:
`L = E_sample [ t_sample * SUM_{a in A(I)}(target[a]-net[a])^2 ]`.
`python/spincore_nn/training.py`, `kind='advantage'`:
`per = SUM_{legal} squared_error / COUNT(legal)`,
seguido de `loss = mean(per * (weight / mean(weight)))`.

Portanto, **há um reescalonamento por 1/|A_legal| que o algoritmo
impresso não especifica**. Para dois estados com igual soma do erro
quadrático e mesma ponderação de amostra, a importância relativa
de estado com 2 ações comparada ao de 7 é multiplicada por **7/2**.
Caso a comparação mantenha erro médio por ação constante, o objetivo
publicado privilegia estados com mais ações enquanto SpinCore os
equilibra. Não confundir com a normalização global dos pesos, que
não é a causa desta divergência.

No relatório do professor @17315, das **367 rows selecionadas e
enviesadas**, os tamanhos legais são:
2:9; 3:65; 4:25; 5:103; 6:124; 7:41.
Isso prova que a variação do denominador ocorre nos estados
auditados; **não** mede sua distribuição na população de treino.

**Juízo: divergência real na geometria do loss, NÃO bug
automaticamente comprovado.** Não afirmar que esta é a
origem dos 4.676 sign-errors sem avaliação de targets; não reescrever
a loss atual nem alterar o contrato do benchmark de quatro objetivos.

### 6. Rede de política média — diferenças declaradas de coleta

`python/spincore/lean_functional_training.py`, função
`_collect_sampled_policy`:
- em trajetórias amostradas, armazena `sigma(I)` para **todo**
  ator encontrado;
- sorteia a ação de cada ator para avançar;
- dá a cada amostra peso `iteration`;
- ao fim, `finalize` ajusta `AveragePolicy` separadamente.

Brown et al. (2019), algoritmo 2: durante travessia external-sampling
do jogador `p`, guarda a estratégia dos **outros** jogadores em
seus nós, que são amostrados, enquanto as ações do traverser são
enumeradas. Isso implementa um mecanismo de reach diferente da
trajetória totalmente amostrada usada no caminho lean.

**Juízo: diferença real da coleta da estratégia média.**
O arquivo `python/spincore/r7_5_action_cfr.py` conserva método
alternativo `collect_strategy_own_reach`, mas a linha lean escolhe
conscientemente outro mecanismo (TRAIN-002, TRAIN-005).
A interpretação de distribuições de treino da AveragePolicy
**não é automaticamente** a estratégia média teórica do Deep CFR.
A política ENS8 pura @17315, contudo, não consulta essa
AveragePolicy; isto **não é explicação direta dos all-ins ENS8**.

### 7. Redes por jogador — desvio de parametrização não é bug de ator

OpenSpiel possui Advantage networks e memories por jogador;
SpinCore possui redes separadas por **domínio HU/3H** e
parametrização compartilhada dos jogadores vivos naquele domínio.
A observação do ator é canonicamente relativa, uma forma de
compartilhar a função entre assentos em jogo simétrico.

**Juízo: adaptação a auditar quanto à sua suficiência, não prova
de erro**. Não abrir novo conjunto de redes por jogador sem
evidência de aliasing/target incompatibility que justifique custo.

## Recomendação técnica, com critério de parada

**NÃO implementar agora** alteração de fallback, agregação,
representação, loss, estratégia média ou arquitetura.
Em particular:
- `G3 published argmax fallback = FAIL`, não reabrir;
- `member-policy mean = direct-play FAIL`, não reabrir;
- regressão de targets de 4.676 posições protegidas sugere
  erro de sinal dos membros, não só agregação; não usar
  a amostra selecionada de 367 como população representativa;
- `semantic_objective_benchmark_17315` continua **READY / PENDING /
  DEFERRED** e não pode ter contrato reescrito depois de definido.

**G4A — prioridade de menor custo, sem retraining:**
confirmar em estados legalmente alcançáveis se há
histórias públicas distintas com o mesmo V1 observável que
requerem estratégias realmente diferentes (não apenas
diferença V2 nominal); se não houver evidência material,
não promover V2. O histórico V2 existe e possibilita comparação.
Sem colisão decisiva, a perda de campos é uma hipótese de
sub-representação, não reparo pronto.

**G4B — prioridade se G4A não apontar defeito material:**
usar a população pré-existente de targets do reservoir e
as próprias métricas protegidas para avaliar a relevância
do fator 1/|A| e a distribuição de discrepâncias de sinal
por número de ações. Isto é diagnóstico read-only;
**não** requer refit ou novas trajetórias CFR.
Somente se variação explicar erros de modo consistente,
propor candidato de loss baseado na fórmula publicada,
sob preregistro próprio — sem reescrever o benchmark
já congelado nem assumir melhora de EV.

**Critérios de parada:** sem defeito demonstrável ou
ganho material associado, manter o baseline, registrar
incerteza e não gastar compute com séries de alternativas
não justificadas. Qualquer arquitetura/representação nova
exige contrato de admissibilidade e preservação de
checkpoints/dados históricos, conforme REP-015/REP-018.

## Limites da auditoria

- leitura estática do main do GitHub, artigo e OpenSpiel; sem execução
  do solver ou carregamento de checkpoints no Ryzen;
- as invariantes do compilador/representação e as
  propriedades de reachability não foram testadas neste gate;
- nenhuma afirmação de força, EV ou superioridade se apoia apenas
  nessas diferenças de fonte.

**Decision: SOURCE_FIDELITY_STATIC_COMPLETE; G4A/G4B READ_ONLY NEXT;
NO_TRAIN; NO_PRODUCTION_PATCH; NO_ARCHITECTURE_SWITCH.**

## Revisão de continuidade após leitura do histórico validado (2026-11-01)

**Correção de prioridade baseada em evidência já existente, sem alterar
contratos ativos:**

1. `validation/R7_5_ARCH_RESET_V1PLUS_CLOSURE_AND_V1_FALLBACK_SELECTION_20260825.md`
   **já documentou** que V1 omite ator/valores no histórico de últimos 32
   `(street,action_type)` tokens. O documento também registra que
   os candidatos V1+ mais ricos **falharam** os gates comuns
   (control TV 0.243976 vs candidate 0.250565, melhoria relativa
   -2.7009%); V1 permaneceu por estabilidade, **não** por
   perfeição estratégica. `REP-018` preserva esse entendimento.
   Logo **não** reabrir no momento um ciclo de experimentos V2/V3
   só porque o limite de dados ficou evidente na fonte.
2. O próprio relatório **já calculado** pelo
   `tools/diagnose_teacher_fallback_sign_error_17315.py` contém
   `by_legal_action_count`, agrupando a população primária de
   **4.676** posições protegidas de target positivo / professor
   em fallback. O contrato congelado de 31/10
   `contracts/teacher_fallback_sign_error_17315_v1.json`
   listava explicitamente `diagnostic by number of legal actions`.
   Portanto, o primeiro passo útil não é uma nova execução:
   é **ler o relatório existente**
   `runs/teacher_fallback_sign_error_17315/teacher_fallback_sign_error_17315.json`.
3. O campo `by_legal_action_count` informa contagens,
   intensidade de target, perda de sinal, votos dos oito modelos.
   **Não** apresenta sozinho contrafactual causal de alterar a
   ponderação da loss. Ele decide se faz sentido investigar
   `1/|A|` com os targets congelados; evita novos testes
   quando a discrepância é pouco informativa.
4. Somente se dados de holdout e análise de regressão indicarem
   que o fator 1/|A| é um contribuinte relevante,
   documentar uma **nova hipótese específica** baseada na
   loss publicada, sem adulterar o benchmark de quatro losses
   já pré-registrado e sem trocar a metodologia por suposição.

**Ordem revisada e justificada:** G4B (revisão do relatório pronto)
**ANTES** de qualquer G4A de trajetória/representação.
G4A só reabre com evidência nova de colisões ou de erro estratégico
material que a seleção de 25/08 ainda não investigou.
Não fazer novo solve, refit, random-seed sweep ou migração para
representação rica neste gate.

Dados já registrados mostram *mecanismo* de amplificação e erro de
sinal, mas **não** identificam causalmente a divisão por |A| como
causa primária. O conjunto protegidos permanece congelado e
não deve ser consultado repetidamente para escolher arquiteturas.

## G4B — análise do relatório de sinal recebido (2026-11-01)

O usuário anexou `SpinCore_teacher_fallback_sign_error_17315.json`;
SHA256 **`e19faacee5cc55e59f94598aeb30435421b707da6a64cfa081c1143140b23405`**,
11.436 bytes. Schema
`SPINCORE_TEACHER_FALLBACK_SIGN_ERROR_17315_V1`, `status=PASS`,
teacher 17315, `protected_holdout_count=50000`, fonte de teacher SHA
`d856acacf4f9d1e03f424c15a1d41cc36bb0384f3cbc62e6fadea84979c6e9f4`.
Os grupos por `by_legal_action_count` somam exatamente 4.676 e a
média ponderada de votos 2.0853293413 reproduz a agregação global.

### Distribuição da população selecionada (não são taxas condicionais)

| Nº de ações legais | Fallback+target positivo | % dos 4.676 | Média de votos positivos no target-best entre 8 redes | Target-best p médio | Bias raw na melhor ação |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2059 | 44.03% | 2.130 | 0.9925 | -0.16786 |
| 3 | 1830 | 39.14% | 1.919 | 0.9122 | -0.18681 |
| 4 | 465 | 9.94% | 2.469 | 0.7707 | -0.19607 |
| 5 | 157 | 3.36% | 2.191 | 0.6704 | -0.14491 |
| 6 | 128 | 2.74% | 2.172 | 0.6299 | -0.13892 |
| 7 | 37 | 0.79% | 2.243 | 0.5919 | -0.19994 |

2–3 ações somam **3.889 / 4.676 = 83,1694%** dos casos
**já selecionados por erro**. O relatório não fornece
`n_target_positive_by_legal_count` em todos os 50.000 protegidos,
portanto **NÃO permite estimar `P(fallback errado | legal_count=k,
target positivo)`**. É inválido concluir que 2–3 ações possuem
a maior taxa de erro sem seus denominadores.
`mean_raw_target_best_bias` é negativo em todos os grupos mas
não segue progressão monotônica simples com `legal_count`.
O tamanho médio do target também varia com `legal_count`.
É **correlação condicionada a um evento selecionado, não causalidade**.

Na população total protegida existem **33.214** targets com
ao menos um Advantage positivo (66,428% do holdout) e 4.676
deles entram em fallback no professor: **14,0784% de erro
condicional global entre target-positivos**, ou 9,352%
de todo holdout. O target-best tem probabilidade média **91,509%**
dentre as discrepâncias e há sinal positivo na ação target-best
em apenas **2,085/8** membros em média. Só **44,67%** dos
4.676 casos ficam nas duas quartilas inferiores de magnitude
de target, portanto o `target_margin_fragile=False` do gate
pré-registrado continua correto. Isso reforça falhas
de aproximação/sinal sem identificar a causa causal.

### Inferência sobre a loss

A função de treinamento atualmente usa
`SUM_{legal}(pred-target)^2 / legal_count`. Isso **aumenta o
peso relativo de estados com 2–3 ações** comparado à
`SUM_{legal}` do algoritmo impresso. A grande proporção de
estados 2–3 na *amostra de erros* não demonstra que a normalização
os penalizou; eles podem ser dominantes entre todos os estados
alcançáveis e target-positivos. Assim, **não há justificativa
técnica suficiente para editar o loss agora**, tampouco para
afirmar que a normalização foi exonerada causalmente.

### Gate terminável, sem nova inferência neural

Foi adicionado `tools/audit_teacher_sign_error_legal_denominators_17315.py`,
**preparado, não executado**, para ler apenas checkpoint RESUME
original, probe do split, e relatório existente; recuperar exatamente
os mesmos 50 mil índices protegidos via `holdout_seed`,
contar `n_target_positive` e `n_total` por número de ações,
verificar `sum(target_positivo)=33214` e calcular taxas
`n_fallback_target_positivo_k / n_target_positivo_k`.
O código valida SHA do resume contra relatório PASS e falha
sem retornar números se houver divergência. **Zero forward
pass, zero CFR root, zero treino, zero alteração em checkpoint,
sem acesso ao sealed holdout**.

Comando PowerShell único:
```powershell
wsl -e bash -lc "cd ~/spincore_lean_functional && git pull --ff-only origin main && .venv_lean/bin/python tools/audit_teacher_sign_error_legal_denominators_17315.py"
```

**Stop rule:** se a taxa condicional não mostrar dependência
convincente de `legal_count`, encerrar a hipótese específica
de normalização e **não gastar compute em refit apenas por ela**.
Se mostrar associação forte, isso **ainda não demonstra causalidade**:
comparar primeiro target-strength e composição de spots e decidir
se o já congelado benchmark supervisionado de objetivos, que não
inclui uma variante pura `SUM_{legal}`, é justificável.
Não adicionar retrospectivamente novo braço a um preregistro
de outubro nem abrir o holdout selado.

## G4B — denominadores executados no Ryzen: PASS (2026-11-01)

O usuário executou o script read-only e colou
`SPINCORE_G4B_LEGAL_DENOMINATORS_PASS`.
A saída valida `resume_state.pt` SHA
`ed0e00a05fe3c6900100b12de5f27efcaf24261adb816090320cc9b34b1d403f`,
`teacher_fallback_sign_error_17315.json` SHA
`e19faacee5cc55e59f94598aeb30435421b707da6a64cfa081c1143140b23405`,
e o split já existente
`runs/3h_advantage_controlled_budget_curve/20260926_031558/3h_advantage_controlled_split_probe.pt`.
Não houve inference forward, raízes CFR, refit ou alteração de pesos.
O output foi registrado como **transcrição derivada do console**
em `contracts/g4b_teacher_sign_error_legal_denominators_17315_v1_result.json`;
o script não salvou um JSON original de saída.

### Distribuição com denominadores CORRETOS

| Ações legais | Todos os 50k | Target positivo | Fallback com target positivo | P(fallback | target+, legal_count) |
|---:|---:|---:|---:|---:|
| 2 | 18.954 | 10.968 | 2.059 | **18,7728%** |
| 3 | 19.388 | 12.870 | 1.830 | **14,2191%** |
| 4 | 5.628 | 4.489 | 465 | **10,3587%** |
| 5 | 2.464 | 1.926 | 157 | **8,1516%** |
| 6 | 2.940 | 2.415 | 128 | **5,3002%** |
| 7 | 626 | 546 | 37 | **6,7766%** |
| **TOTAL** | **50.000** | **33.214** | **4.676** | **14,0784%** |

Agrupamento pré-descritivo: 2–3 ações **3.889/23.838 = 16,3143%**
versus 4–7 ações **787/9.376 = 8,3938%**; razão
**1,9436**, diferença absoluta **7,9205 pp**.
Existe associação relevante: taxas decrescem de 2 a 6, com
pequena reversão em 7 (somente 37 eventos). Não concluir
causa nem precisão independente a partir de trajetórias correlacionadas.

### Checagem causal com o código do resolvedor

`src/lean_action_abstraction.cpp` demonstra que legal_count é
endógeno à situação de jogo:
- pré-flop com >=2 raises elimina opções fracionárias;
  com >=3 raises pode proibir `CHECK_CALL` e restringir
  decisão a fold/shove;
- all-in puramente equivalente a call não é duplicado;
- `d.got_isolated` pode suprimir raises;
- pós-flop, `min_raise_to`, recursos restantes, apostas
  que envolvem >=60% do stack efetivo, e deduplicação
  eliminam alguns tamanhos de aposta.

Portanto as taxas maiores com 2–3 ações **podem refletir
situações de decisão/facing-raise mais difíceis**,
além do número de ações. O JSON agregado não identifica
o efeito da perda de informação histórica, do número de
ações, nem da normalização do loss controlando
posição/street/pot/to_call/mão de cada estado.

O objetivo MSE atual **já** aumenta, relativamente ao
`SUM_{legal}` publicado, a importância de estados com
poucas ações; logo não há evidência direta de que
alterar `MEAN_{legal}` para `SUM_{legal}` corrigiria o
grupo com maiores taxas. **NÃO** editar loss ou
acrescentar braço SUM retrospectivamente.

### Decisão após G4B

**G4B PASS como evidência descritiva, CAUSALITY NOT ESTABLISHED.**

A revisão estrutural e os gates de fallback/aggregation anteriores
não localizaram uma correção pontual segura; o problema de
sinal do ENS8 persiste. É racional agora passar ao **único
experimento supervisionado limitado já definido e congelado**,
`contracts/semantic_objective_benchmark_17315_v1.json`,
**sem modificar os quatro braços**: A = refit idêntico
ao baseline, B = escala normalizada, C = balanceamento
positive/nonpositive, D = normalizado+balanceado.
Essa experiência pode determinar se objetivos de
aproximação recuperam o sinal no reservoir existente,
**não** testar diretamente o efeito de `1/|A|`.

Manifesto `contracts/run_manifests/semantic_objective_benchmark_17315.json`
validado estaticamente em 2026-11-01: os **dez** blob SHAs,
inclusive runner, correspondem aos arquivos atuais no GitHub.
`status=READY`, `block_reason=null`.
A execução de 4 x ENS8 refits com 1.600 passos por membro
(**32 fits, 51.200 passos de otimização**) é computação
supervisionada de escopo limitado, e **não** continuação do
treinamento CFR nem geração de novas mãos/targets.
O runner exige pelo menos 22 GiB WSL disponíveis,
árvore git limpa, treino longo ausente e checagem de
paridade bit a bit do braço A com @17315, antes das
alternativas. Falha em qualquer requisito deve
**parar**, sem fazer workaround experimental.

Comando PowerShell (o usuário decide se executa):
```powershell
wsl -e bash -lc "cd ~/spincore_lean_functional && git pull --ff-only origin main && bash tools/run_semantic_objective_benchmark_17315.sh"
```

**Critério de parada seguinte:** se nenhum B/C/D
cumprir todos os gates congelados,
não reabrir seeds, arquitetura, loss ou holdout
à procura de um PASS; registrar falha e reconsiderar
a estratégia global de treinamento em um gate
separado. Se algum passar, somente autorizar
próximo gate de avaliação de patologias/jogo direto,
não produção nem treino longo. Preservar
@10105, @17315, todos os reservoirs e snapshots.
