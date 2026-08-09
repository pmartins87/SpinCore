#include "test_framework.hpp"
int main(){int pass=0,fail=0;for(auto&t:spin_tests()){try{t.fn();++pass;std::cout<<"PASS "<<t.name<<"\n";}catch(const std::exception&e){++fail;std::cerr<<"FAIL "<<t.name<<": "<<e.what()<<"\n";}catch(...){++fail;std::cerr<<"FAIL "<<t.name<<": unknown\n";}}std::cout<<pass<<" PASS / "<<fail<<" FAIL\n";return fail?1:0;}
