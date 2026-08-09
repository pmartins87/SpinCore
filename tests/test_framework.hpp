#pragma once
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
struct SpinTestCase{std::string name;std::function<void()>fn;};inline std::vector<SpinTestCase>& spin_tests(){static std::vector<SpinTestCase>x;return x;}struct SpinRegistrar{SpinRegistrar(const char*n,std::function<void()>f){spin_tests().push_back({n,std::move(f)});}};
#define SPIN_TEST(name) static void name(); static SpinRegistrar reg_##name(#name,name); static void name()
#define REQUIRE(x) do{if(!(x))throw std::runtime_error(std::string("REQUIRE failed: ")+ #x + " @"+__FILE__+":"+std::to_string(__LINE__));}while(0)
#define REQUIRE_THROWS(x) do{bool threw=false;try{(void)(x);}catch(...){threw=true;}if(!threw)throw std::runtime_error(std::string("REQUIRE_THROWS failed: ")+ #x);}while(0)
