#pragma once
#include <cstdint>
namespace spincore {
enum class ExactActionType : std::uint8_t { Fold=0, Check=1, Call=2, BetTo=3, RaiseTo=4, AllIn=5 };
struct ExactAction { ExactActionType type{ExactActionType::Check}; std::int32_t amount_to{0}; friend bool operator==(const ExactAction&,const ExactAction&)=default; };
enum class AbstractActionSlot : std::uint8_t { Fold=0, CheckCall=1, ContextRaise=2, SmallPot=3, LargePot=4, AllIn=5 };
[[nodiscard]] const char* action_name(AbstractActionSlot a) noexcept;
}
