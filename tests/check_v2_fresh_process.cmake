if(NOT DEFINED PROBE)
  message(FATAL_ERROR "PROBE executable not supplied")
endif()

execute_process(
  COMMAND "${PROBE}"
  RESULT_VARIABLE r1
  OUTPUT_VARIABLE o1
  ERROR_VARIABLE e1
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT r1 EQUAL 0)
  message(FATAL_ERROR "first V2 probe failed (${r1}): ${e1}")
endif()

execute_process(
  COMMAND "${PROBE}"
  RESULT_VARIABLE r2
  OUTPUT_VARIABLE o2
  ERROR_VARIABLE e2
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT r2 EQUAL 0)
  message(FATAL_ERROR "second V2 probe failed (${r2}): ${e2}")
endif()

if(NOT o1 STREQUAL o2)
  message(FATAL_ERROR "SPNNIV2 fresh-process payload mismatch")
endif()

string(LENGTH "${o1}" hex_length)
# 830 bytes serialized as two hex characters each.
if(NOT hex_length EQUAL 1660)
  message(FATAL_ERROR "unexpected SPNNIV2 probe hex length: ${hex_length}")
endif()

message(STATUS "SPNNIV2 fresh-process equivalence PASS (${hex_length} hex chars)")
