# SpinCore — Auditoria de raiz: Advantage cru do professor ENS8 @17315 (2026-11-01)

Status: **RECEBIDO / INDEPENDENTEMENTE RECOMPUTADO / SEM AUTORIZAR ALTERAÇÃO**.
Fonte operacional: `CURRENT_WORK.md`; decisões e limites: `ROADMAP.md`.

## Dados e integridade

Arquivo original enviado ao chat: `SpinCore_teacher_raw_advantage_17315(1).json`.
SHA256 **`2d57cc58343948101e6d9be4e91aaa70392da6f02e15e2b09ece0d5ba5c4c456`**;
1.661.527 bytes.
Schema: `SPINCORE_3H_TEACHER_RAW_ADVANTAGE_DIAGNOSTIC_17315_V1`;
status PASS; teacher 17315; 367 rows; 367 unique target keys; zero
`selection_mismatches`, zero `missing_keys`; 198 positive-regret and
169 softmax-fallback. Original large report stays no anexo, não regravado
no repositório.

Uma rotina Python independente refez para todas as 367 rows:
(1) média aritmética dos oito valores crus por ação legal;
(2) regret matching dos positivos e fallback softmax quando todos <=0;
(3) regret matching/fallback **por membro**, depois média das oito políticas;
(4) comparação com `sigma`, `mean_raw_advantages` e
`member_policy_mixture_selected_probability` do JSON.

Erros absolutos máximos: **0.0** na sigma final, **0.0** na
média raw, **1.11e-16** na média das políticas (arredondamento).
Isso mostra **cálculo reproduzível**, não demonstra validade dos labels CFR
ou qualidade de jogo.

## G4-diagnóstico: mecanismo de saturação de política explicado

Nas 20 rows selecionadas no G2 como **high-card/no-immediate-draw top probability**
sob o teacher de base:
- média **p(ALL_IN)=96.9550%** para `RM(mean(raw))`;
- média **p(ALL_IN)=37.5314%** para `mean(RM(raw_member))`;
- 20/20 p(ALL_IN)>=80% em RM(mean); nenhuma >=80% na média das políticas;
- 20/20 têm amplificação >=25pp; 13/20 sem maioria de argmax por membro
  (menos de quatro dos oito preferem ALL_IN); 15/20 são **ALL_IN o único
  Advantage médio positivo**.
- `mean(raw_ensemble_ALL_IN)` varia aproximadamente **0.0004023 a 0.0093869**;
  a magnitude do número positivo não impede p=100% se os outros
  valores médios são não positivos.
- As 20 observações são **19 estados de cartas/aposta únicos** (um estado
  aparece em lineups distintos). Não são amostra populacional aleatória.
  Os resultados de média não devem ser interpretados como frequência global.

Exemplo mais explícito (scenario 3980, lineup 3, decision 7):
- high-card sem draw, stack efetivo aprox. 20.7bb, `to_call=0`,
  pot=148, `hole=[38,51]`, `board=[3,24,46,5]`;
- média raw `ALL_IN=+0.0004022980574518442`;
  `CHECK_CALL=-0.0013085066`, outras quatro ações também negativas;
- `RM(mean(raw)) => p(ALL_IN)=100%`;
- somente **1/8** membros tem ALL_IN como maior p individual;
  `mean(policy_members) => p(ALL_IN)=21.0437%`.
A discrepância é aritmética e derivada do `max(raw,0)/sum(max(raw,0))`.
Regret matching é aplicado corretamente, mas previsões próximas de zero,
erros de sinal e **ordem de agregação não linear** podem provocar
política saturada. Não atribuir a diferença ao fallback.

Em amostra 271 sampled high-card-jam rows:
- 142 RM positivo; 129 fallback; p(ALL_IN) média
  **44.1096%** com raw-mean vs **25.7255%** na média das políticas;
- 72/271 têm amplificação >=25pp; 26/271 são >=80% sem maioria
  dos membros; amostra condicionada a flags do teacher, sujeita a viés.

## Mecanismo separado: 72o deep

Nas 18 rows sampled `PREFLOP_DEEP_72O_JAM`, **18/18** usam
`SOFTMAX_FALLBACK`, com p(ALL_IN) média **33.1805%** (quase
1/3 quando três ações legais). Na superfície top20, 20/20 fallback,
p(ALL_IN) **33.2614%**. O fallback publicado Deep CFR usa
argmax do raw se nenhum regret legal positivo, em vez de softmax.
Entretanto G3 de 10k cenários **reprovou** a substituição isolada por
não atingir o limiar de redução de 72o pre-registrado. Não rever
limites ex post.

## Relação com a evidência precedente

**Não é uma descoberta inédita ou um novo gate com custo de execução:**
`CURRENT_WORK.md` já documentou 69.68% dos estados com
fallback do teacher apresentando target CFR protegido com regret positivo.
Dos **4,676** estados discrepantes auditados anteriormente:
- em 18.63%, **nenhum** dos oito membros preservou sinal positivo
  na ação preferida pelo target;
- apenas 18.63% tinham >=4 membros preservando tal sinal;
- média 2.085/8 membros com sinal correto;
- target strength média 0.1590 e mediana 0.1493;
- diagnóstico anterior `MIXED_MODEL_AND_ENSEMBLE_FAILURE`,
  `target_margin_fragile=False`.

**Limite:** este arquivo raw de 367 estados contém previsões, NÃO
contém os Advantage targets CFR corretos desses mesmos estados.
Ele comprova a origem algébrica da saturação de p, mas **não prova que
o target verdadeiro favorecia fold/check**, nem determina se a
discrepância específica se deve a input observacional, capacidade
da rede, pesos de loss, dados ou ruído dos targets. A evidência de
erro de sinal em outro conjunto vem do diagnóstico protegido anterior.

Benchmarks anteriores já fecharam a opção fácil de
`mean(member policy)`: candidate B perdeu o gate de direct play
em 2026-10-31 e piorou a frequência de 72o. Não reintroduzir
essa alternativa como solução de produção só porque 37.53% < 96.96%.

## Comparação com procedimento publicado

O Brown et al. 2019 Deep CFR e a implementação OpenSpiel possuem
Advantage redes por jogador e rede de estratégia média separada,
com regret matching da rede corrente e argmax no fallback.
A política SpinCore de inference direta `mean(raw oito membros @17315)
-> regret matching` é uma **adaptação própria**; não é o algoritmo
temporal completo Deep CFR ou SD-CFR de referência.

Fontes primárias:
- https://proceedings.mlr.press/v97/brown19b.html
- https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/deep_cfr.py

A prova teórica para 2 jogadores soma zero não certifica convergência
em Spin & Go 3-handed; evidência publicada em poker não constitui gate
de performance do nosso ambiente.

## Decisão / plano fechado

1. **Manter** rede, ENS8, fallback, checkpoints, reservoirs e produção
   exatamente como estão. Sem novo treino nem direct-play.
2. **Próximo passo read-only, usando código/dados existentes**: confrontar
   `MIXED_MODEL_AND_ENSEMBLE_FAILURE` no holdout protegido com
   semântica da observação, identidade/posição do jogador, máscaras legais
   e caminhos de preparação dos targets. Verificar fidelidade
   com algoritmo de referência por jogador / separate average strategy.
3. Se houver **defeito implementacional comprovado**, preparar correção
   mínima isolada com parity/regressão e gate econômico; não aplicar
   automaticamente ao baseline. Se não houver, considerar o já preparado
   benchmark supervisionado das losses com justificativa de referência
   científica, ou uma extração temporal da estratégia já preservada.
   Nem reativar treinos longos nem promover candidatos sem novo
   benefício direto e critérios registrados antes.
4. **Parada**: não repetiremos benchmarks de aggregation/fallback
   sem nova hipótese testável ou evidência contraditória relevante.

A auditoria explica o mecanismo, mas **não recupera EV ainda**.
