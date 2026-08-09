#pragma once
#include "spincore/canonical_infoset.hpp"
#include "spincore/hand_engine.hpp"
namespace spincore { [[nodiscard]] CanonicalInfoset build_current_actor_infoset(const HandEngine& hand,std::int32_t blind_index); }
