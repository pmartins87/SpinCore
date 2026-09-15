# SpinCore — poker goals and what is being trained

Status: **CANONICAL POKER-LEVEL DESCRIPTION**
Date: 2026-09-15

This document deliberately explains SpinCore in poker terms rather than computer-science terms.

## 1. What SpinCore is being trained to do

SpinCore is being trained to make complete No-Limit Hold'em decisions in Spin & Go states, from preflop through river, in both three-handed play and true heads-up.

It is not being taught one fixed chart and it is not being taught one fixed stack depth. The training repeatedly puts the player into realistic tournament situations with different blinds, stacks, positions, hole cards, boards and prior betting actions and asks a poker question:

**Which legal action wins the most chips on average from this situation?**

The current first-release objective is chip EV. A good decision is one that improves long-run tournament-chip expectation, not one that merely wins the current pot more often.

## 2. Tournament situations it sees

The training contains both major Spin & Go phases:

- three players still alive;
- true heads-up after one player has been eliminated.

It uses the historical real-hand distribution recovered from DeepSpin rather than pretending every hand is played at 10/20.

Blind levels represented are:

- 10/20;
- 15/30;
- 20/40;
- 30/60;
- 40/80;
- 50/100;
- 60/120;
- 80/160;
- 100/200.

Early levels are naturally much more common in three-handed play. Heads-up occurs much more frequently in the middle and later blind levels. The training follows those observed proportions.

Stacks are not fixed. For each blind level the player is placed into stack configurations derived from the historical SpinGo data. Therefore SpinCore sees deep, medium, shallow and very shallow decisions rather than learning a single effective-stack strategy.

The tournament has 1500 chips in play. In heads-up, one seat is dead and the remaining 1500 chips are split between the two players. In three-handed play the 1500 chips are split among all three players.

## 3. Positions

SpinCore must learn position-specific poker.

Three-handed situations include the relationships among BTN, SB and BB. Heads-up includes both button/small-blind and big-blind play. Dealer position is varied rather than fixed.

This means the same hand is not supposed to have one universal answer. For example, an offsuit ace on the button with 20 BB effective is a different poker decision from the same cards in the big blind facing aggression.

## 4. Preflop poker being learned

The current action vocabulary deliberately preserves the mature DeepSpin/Crusher-style practical sizing family instead of giving the learner dozens of nearly identical raise sizes.

Preflop it can learn among:

- fold;
- check when checking is possible;
- call/limp;
- normal open to 2 BB;
- isolation raise after limpers, beginning at 2.5 BB and increasing with extra limpers;
- normal 3-bet family, beginning at 5 BB and increasing when callers entered after the initial raise;
- all-in.

The tree also preserves practical poker restrictions from the mature legacy strategy. For example, after enough preflop aggression it stops offering a collection of meaningless small re-raises and the meaningful decision becomes closer to fold/call/shove. A limper facing an isolation also follows the legacy context instead of being treated as an unopened-pot raiser.

What is being learned is **when** to use those actions and with which hand/position/stack/context — not the numerical sizes themselves.

So SpinCore should eventually learn things such as:

- opening ranges by position and stack depth;
- limp versus raise frequencies where appropriate;
- isolation ranges;
- calling versus open ranges;
- 3-bet and shove ranges;
- defense against 3-bets and isolation raises;
- short-stack push/fold transitions;
- how all those ranges change between three-handed and heads-up.

## 5. Postflop poker being learned

After the flop SpinCore can choose among:

- fold;
- check/call;
- bet or raise roughly 33% pot;
- 50% pot;
- 75% pot;
- 100% pot;
- all-in.

The percentage is based on the pot after accounting for a call when Hero is facing a bet. Illegal or nonsensical sizes are removed. Bets that are effectively committing the stack are collapsed toward all-in rather than creating artificial tiny stacks behind.

SpinCore therefore has to learn poker concepts implicitly from results rather than from a hardcoded sentence such as "top pair always calls". Across many situations it should learn when to:

- continuation bet or check back;
- defend versus continuation bets;
- value bet thinly or strongly;
- bluff and semi-bluff;
- protect vulnerable made hands;
- realize equity by checking/calling;
- raise draws;
- slow-play very strong hands when appropriate;
- fold dominated or low-equity holdings;
- change betting size according to board, stack, position, range interaction and previous action;
- continue correctly from flop to turn and river instead of treating each street in isolation.

## 6. Cards and boards

SpinCore sees exact hole cards and exact public cards. It is not trained only on hand labels such as "top pair" or "two pair".

This matters because one of the historical DeepSpin failures was confusing a board-made hand with strength genuinely contributed by Hero's cards. The current direction avoids making a fragile hand-class label the sole description of the poker state.

Therefore boards such as paired boards, monotone boards, connected boards, ace-high boards, dry boards and draw-heavy boards are all represented through the actual cards and action history.

## 7. What the opponents are during training

The main training is self-play. SpinCore is not trained only against a passive fish, only against a maniac, or only against DeepCrusher.

During learning, the opponents come from the evolving strategy itself. This forces the policy to respond to a broad range of reasonable actions rather than memorize how to exploit one fixed bot.

That is important for the final objective: SpinCore should become a generally strong poker player, not merely a DeepCrusher counter-strategy.

DeepCrusher is therefore an **evaluation opponent and benchmark**, not the sole teacher.

## 8. What winning means during training

The current strategic objective is chip EV.

If two legal lines are compared from the same situation, the better line is the one that produces more tournament chips on average over repeated deals and opponent responses.

The learner is therefore not rewarded merely because:

- it won the current hand;
- it reached showdown;
- it avoided variance;
- it survived longer;
- it made a conventional-looking action.

A profitable bluff that often loses the individual pot can be correct. A call that wins the hand frequently can still be bad if it loses too many chips when wrong.

## 9. Why there are separate 3H and HU strategies

Three-handed and heads-up poker are strategically different enough that SpinCore keeps separate brains for them.

Heads-up has much wider ranges, more blind pressure and no third player's range or stack to consider. Three-handed decisions depend heavily on which two positions are interacting and on the presence of the third stack/player.

The goal is not to force one compromise strategy to handle both formats.

## 10. What is not yet the target

The first-release training does not yet try to optimize separate policies for rare multi-place payout structures. The main policy is WTA/chip-EV based.

It also does not train specifically for one monetary buy-in. A 10/20 blind situation with the same stack geometry is poker-equivalent whether the tournament buy-in is small or large, assuming the game rules are identical.

## 11. DeepCrusher benchmark — mandatory acceptance metric

A major product-quality metric is now explicit:

**SpinCore must beat DeepCrusher in an extensive, fair offline simulation under the common game semantics.**

This benchmark must not be a cherry-picked hand sample. It should use:

- common deals wherever possible;
- seat/position rotation;
- both HU and three-handed play when the common strategy contract supports it;
- the full realistic blind/stack distribution rather than 10/20 only;
- enough hands/scenarios that random card variance is small relative to the measured edge;
- chip-EV reporting overall and by domain/blind/position;
- confidence intervals or another clear uncertainty estimate;
- no hidden special knowledge of the opponent's cards or future board;
- the same legal game engine for both policies.

The minimum success condition is not "won one sample". SpinCore should show a statistically convincing positive edge over DeepCrusher in the broad benchmark, and that edge should not come only from one narrow blind level or seat.

### Direct HU benchmark

HU is the cleanest head-to-head comparison. SpinCore and DeepCrusher receive the same deals with seats swapped across paired repetitions. Report total chips won/lost and chip EV per hand, with blind-level detail.

### Three-handed benchmark

Three-handed comparison must avoid giving one bot an accidental seat/composition advantage. Rotate seats and use mirrored lineups, for example SpinCore against two DeepCrusher instances and the reciprocal composition, or another paired design that gives each policy equal exposure to every seat and opponent composition.

### Full-match/tournament benchmark

When the continuous tournament-transition model is finalized, add a second benchmark that plays complete Spin & Go matches from three-handed start through heads-up and elimination. That benchmark should report match/tournament win rate in addition to chip EV. Until that transition model is explicitly defined, hand/state-based paired simulation remains the more controlled measure.

DeepCrusher victory is a **necessary benchmark, not the definition of optimal play**. A policy could overfit specifically to DeepCrusher and still be weak against stronger or different opponents. Therefore DeepCrusher sits beside broader fixed-opponent, self-play and future exploitability/solver-oriented checks.

## 12. Product goal in poker language

The intended end state is straightforward:

SpinCore should be able to sit in realistic three-handed or heads-up Spin & Go situations across the actual blind ladder, understand its cards, position, effective stacks, pot and betting history, and choose strong preflop, flop, turn and river actions that maximize long-run chip expectation.

It should not merely "look reasonable". It should demonstrate its strength by beating credible reference strategies — with DeepCrusher as one of the mandatory reference opponents — over enough paired simulation that variance is no longer a plausible explanation for the result.
