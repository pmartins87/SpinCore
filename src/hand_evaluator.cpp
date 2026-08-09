#include "spincore/hand_evaluator.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
namespace spincore {
namespace {
std::uint8_t straight_high(const std::array<int,15>& count) {
    for (int hi=14; hi>=5; --hi) {
        bool ok=true; for (int r=hi; r>hi-5; --r) if (!count[r]) { ok=false; break; }
        if (ok) return static_cast<std::uint8_t>(hi);
    }
    if (count[14] && count[5] && count[4] && count[3] && count[2]) return 5;
    return 0;
}
}
HandRank evaluate_five(const std::array<Card,5>& cards) {
    std::array<int,15> cnt{}; std::array<int,4> suits{};
    for (auto c: cards) { if (!c.valid()) throw std::invalid_argument("invalid card in hand"); ++cnt[c.rank]; ++suits[c.suit]; }
    bool flush = std::any_of(suits.begin(), suits.end(), [](int x){return x==5;});
    const auto sh = straight_high(cnt);
    if (flush && sh) return {HandCategory::StraightFlush,{sh,0,0,0,0}};
    int quad=0, trip=0; std::vector<int> pairs;
    for (int r=14;r>=2;--r) { if (cnt[r]==4) quad=r; else if (cnt[r]==3 && !trip) trip=r; else if (cnt[r]>=2) pairs.push_back(r); }
    if (quad) { int k=14; while (k==quad || !cnt[k]) --k; return {HandCategory::Quads,{(std::uint8_t)quad,(std::uint8_t)k,0,0,0}}; }
    if (trip && !pairs.empty()) return {HandCategory::FullHouse,{(std::uint8_t)trip,(std::uint8_t)pairs.front(),0,0,0}};
    if (flush) { HandRank h{HandCategory::Flush,{}}; int j=0; for (int r=14;r>=2;--r) for(int n=0;n<cnt[r];++n) h.kickers[j++]=(std::uint8_t)r; return h; }
    if (sh) return {HandCategory::Straight,{sh,0,0,0,0}};
    if (trip) { HandRank h{HandCategory::Trips,{(std::uint8_t)trip,0,0,0,0}}; int j=1; for(int r=14;r>=2;--r) if(r!=trip&&cnt[r]) h.kickers[j++]=(std::uint8_t)r; return h; }
    if (pairs.size()>=2) { int p1=pairs[0],p2=pairs[1],k=14; while(k==p1||k==p2||!cnt[k])--k; return {HandCategory::TwoPair,{(std::uint8_t)p1,(std::uint8_t)p2,(std::uint8_t)k,0,0}}; }
    if (pairs.size()==1) { int p=pairs[0]; HandRank h{HandCategory::Pair,{(std::uint8_t)p,0,0,0,0}}; int j=1; for(int r=14;r>=2;--r) if(r!=p&&cnt[r]) h.kickers[j++]=(std::uint8_t)r; return h; }
    HandRank h{HandCategory::HighCard,{}}; int j=0; for(int r=14;r>=2;--r) if(cnt[r]) h.kickers[j++]=(std::uint8_t)r; return h;
}
HandRank evaluate_best(std::span<const Card> cards) {
    if (cards.size()<5 || cards.size()>7) throw std::invalid_argument("evaluate_best requires 5..7 cards");
    HandRank best{}; bool first=true;
    for (std::size_t a=0;a<cards.size()-4;++a) for(std::size_t b=a+1;b<cards.size()-3;++b)
    for(std::size_t c=b+1;c<cards.size()-2;++c) for(std::size_t d=c+1;d<cards.size()-1;++d)
    for(std::size_t e=d+1;e<cards.size();++e) {
        std::array<Card,5> x{cards[a],cards[b],cards[c],cards[d],cards[e]}; auto r=evaluate_five(x);
        if(first || r>best){best=r;first=false;}
    }
    return best;
}
}
