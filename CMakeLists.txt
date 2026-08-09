cmake_minimum_required(VERSION 3.20)
project(SpinCore VERSION 1.12.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(spincore STATIC
    src/card.cpp
    src/hand_evaluator.cpp
    src/hand_semantics.cpp
    src/ruleset_contract.cpp
    src/scenario_sampler.cpp
    src/game_topology.cpp
    src/action_abstraction.cpp
    src/canonical_infoset.cpp
    src/neural_encoder.cpp
    src/betting_engine.cpp
    src/hand_engine.cpp
    src/hand_infoset_adapter.cpp
    src/tournament_value.cpp
    src/spin_traversal_state.cpp
)

target_include_directories(spincore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(spincore PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(spincore PRIVATE /W4 /permissive-)
else()
    target_compile_options(spincore PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_library(spincore_solver_c SHARED src/solver_c_api.cpp)
target_link_libraries(spincore_solver_c PRIVATE spincore)
target_include_directories(spincore_solver_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(spincore_solver_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()

add_executable(spincore_tests
    tests/test_main.cpp
    tests/test_hand_evaluator.cpp
    tests/test_hand_semantics.cpp
    tests/test_ruleset_contract.cpp
    tests/test_scenario_sampler.cpp
    tests/test_game_topology.cpp
    tests/test_action_abstraction.cpp
    tests/test_canonical_infoset.cpp
    tests/test_neural_encoder.cpp
    tests/test_betting_engine.cpp
    tests/test_hand_engine.cpp
    tests/test_hand_infoset_adapter.cpp
    tests/test_tournament_value.cpp
    tests/test_spin_traversal_state.cpp
    tests/test_solver_frontier.cpp
)
target_link_libraries(spincore_tests PRIVATE spincore spincore_solver_c)
add_test(NAME spincore_tests COMMAND spincore_tests)

add_executable(spincore_scenario_dump tools/scenario_dump.cpp)
target_link_libraries(spincore_scenario_dump PRIVATE spincore)

add_executable(spincore_validate_icm tools/validate_icm_rollout.cpp)
target_link_libraries(spincore_validate_icm PRIVATE spincore)
add_executable(spincore_validate_evaluator tools/validate_evaluator_exhaustive.cpp)
target_link_libraries(spincore_validate_evaluator PRIVATE spincore)
