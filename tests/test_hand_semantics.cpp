#include "test_framework.hpp"
#include "spincore/hand_semantics.hpp"
using namespace spincore;
SPIN_TEST(hand_class_pair){REQUIRE(hand_class({Card{14,0},Card{14,2}})=="AA");}
SPIN_TEST(hand_class_suited){REQUIRE(hand_class({Card{13,1},Card{14,1}})=="AKs");}
SPIN_TEST(hand_class_offsuit){REQUIRE(hand_class({Card{14,0},Card{13,1}})=="AKo");}
