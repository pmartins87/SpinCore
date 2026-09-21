#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

std::map<std::string,double> g_symbols;
std::string g_hand="MOCK-000001";

void set_initial_hu_frame() {
    g_symbols.clear();
    g_symbols["userchair"]=0;
    g_symbols["dealerchair"]=0;
    g_symbols["betround"]=1;
    g_symbols["sblind"]=10;
    g_symbols["bblind"]=20;
    g_symbols["playersdealtbits"]=(1<<0)|(1<<5);
    g_symbols["playersplayingbits"]=(1<<0)|(1<<5);
    g_symbols["playersallinbits"]=0;
    g_symbols["pot"]=30;
    g_symbols["ncommoncardsknown"]=0;

    for (int chair=0;chair<10;++chair) {
        g_symbols["balance"+std::to_string(chair)]=0;
        g_symbols["currentbet"+std::to_string(chair)]=0;
    }
    g_symbols["balance0"]=740;
    g_symbols["currentbet0"]=10;
    g_symbols["balance5"]=730;
    g_symbols["currentbet5"]=20;

    // Hero AsKh. OpenHoldem suit constants:
    // hearts=0, diamonds=1, clubs=2, spades=3.
    g_symbols["$$pr0"]=14;
    g_symbols["$$ps0"]=3;
    g_symbols["$$pr1"]=13;
    g_symbols["$$ps1"]=0;

    for (int i=0;i<5;++i) {
        g_symbols["$$cr"+std::to_string(i)]=0;
        g_symbols["$$cs"+std::to_string(i)]=0;
    }
}

template<typename T>
T load_proc(HMODULE dll,const char* name) {
    const auto p=GetProcAddress(dll,name);
    if (!p) throw std::runtime_error(std::string("missing DLL export ")+name);
    return reinterpret_cast<T>(p);
}

void write_json(
    const std::filesystem::path& report,
    const char* verdict,
    double slot,
    double exact_type,
    double exact_amount,
    double domain,
    double prob_sum,
    int legal_count,
    bool cache_stable,
    bool action_barrier) {
    std::ofstream out(report);
    if (!out) throw std::runtime_error("cannot write mock-host report");
    out<<"{\n"
       <<"  \"schema\": \"SPINCORE_LT2_OPENHOLDEM_SHADOW_DLL_MOCK_HOST_V1\",\n"
       <<"  \"verdict\": \""<<verdict<<"\",\n"
       <<"  \"shadow_only\": true,\n"
       <<"  \"host_api_resolved\": true,\n"
       <<"  \"hand_anchor_started\": true,\n"
       <<"  \"myturn_shadow_decision_ready\": true,\n"
       <<"  \"action_slot\": "<<slot<<",\n"
       <<"  \"exact_type\": "<<exact_type<<",\n"
       <<"  \"exact_amount_to\": "<<exact_amount<<",\n"
       <<"  \"domain\": "<<domain<<",\n"
       <<"  \"probability_sum\": "<<prob_sum<<",\n"
       <<"  \"legal_action_count\": "<<legal_count<<",\n"
       <<"  \"repeated_myturn_cache_stable\": "<<(cache_stable?"true":"false")<<",\n"
       <<"  \"action_queries_hard_zero\": "<<(action_barrier?"true":"false")<<",\n"
       <<"  \"table_actions_executed\": 0\n"
       <<"}\n";
}

} // namespace

extern "C" __declspec(dllexport) double GetSymbol(const char* name) {
    if (!name) return 0.0;
    const auto it=g_symbols.find(name);
    return it==g_symbols.end()?0.0:it->second;
}

extern "C" __declspec(dllexport) char* GetHandnumber() {
    return g_hand.data();
}

extern "C" __declspec(dllexport) void WriteLog(char* format,va_list args) {
    if (!format) return;
    std::vfprintf(stderr,format,args);
}

int wmain(int argc,wchar_t** argv) {
    try {
        if (argc!=3) {
            throw std::runtime_error(
                "usage: spincore_lt2_openholdem_shadow_dll_mock_host DLL_PATH REPORT_PATH");
        }
        const std::filesystem::path dll_path=argv[1];
        const std::filesystem::path report=argv[2];
        set_initial_hu_frame();

        HMODULE dll=LoadLibraryW(dll_path.c_str());
        if (!dll) {
            throw std::runtime_error(
                "LoadLibrary failed, Win32 error="+std::to_string(GetLastError()));
        }

        using Update=void(__stdcall*)();
        using Query=double(__stdcall*)(const char*);
        const auto new_formula=load_proc<Update>(dll,"DLLUpdateOnNewFormula");
        const auto connection=load_proc<Update>(dll,"DLLUpdateOnConnection");
        const auto handreset=load_proc<Update>(dll,"DLLUpdateOnHandreset");
        const auto heartbeat=load_proc<Update>(dll,"DLLUpdateOnHeartbeat");
        const auto myturn=load_proc<Update>(dll,"DLLUpdateOnMyTurn");
        const auto query=load_proc<Query>(dll,"ProcessQuery");

        new_formula();
        connection();
        handreset();
        heartbeat();
        myturn();

        const bool shadow_only=query("dll$spincore_shadow_only")==1.0;
        const bool loaded=query("dll$spincore_loaded")==1.0;
        const bool active=query("dll$spincore_hand_active")==1.0;
        const bool failed=query("dll$spincore_failed")!=0.0;
        const bool ready=query("dll$spincore_ready")==1.0;

        const double slot=query("dll$spincore_action_slot");
        const double exact_type=query("dll$spincore_exact_type");
        const double exact_amount=query("dll$spincore_exact_amount_to");
        const double domain=query("dll$spincore_domain");
        const double first_index=query("dll$spincore_decision_index");
        const double first_u=query("dll$spincore_sample_u");
        const double first_inference=query("dll$spincore_inference_count");

        double prob_sum=0.0;
        int legal_count=0;
        for (int i=0;i<10;++i) {
            const std::string p="dll$spincore_p"+std::to_string(i);
            const std::string l="dll$spincore_l"+std::to_string(i);
            const double prob=query(p.c_str());
            const double legal=query(l.c_str());
            if (prob<0.0 || prob>1.0) throw std::runtime_error("probability outside 0..1");
            if (legal!=0.0 && legal!=1.0) throw std::runtime_error("legal mask outside {0,1}");
            if (legal==0.0 && prob!=0.0) throw std::runtime_error("illegal action has probability mass");
            prob_sum+=prob;
            if (legal==1.0) ++legal_count;
        }

        myturn();
        const bool cache_stable=
            query("dll$spincore_decision_index")==first_index &&
            query("dll$spincore_sample_u")==first_u &&
            query("dll$spincore_inference_count")==first_inference;

        heartbeat();
        myturn();
        const bool duplicate_stable=
            query("dll$spincore_decision_index")==first_index &&
            query("dll$spincore_sample_u")==first_u &&
            query("dll$spincore_inference_count")==first_inference;

        bool action_barrier=true;
        for (const char* q:{
            "dll$fold","dll$check","dll$call","dll$rais",
            "dll$alli","dll$betsize","dll$deep_action"}) {
            action_barrier=action_barrier && query(q)==0.0;
        }

        const bool pass=
            shadow_only && loaded && active && !failed && ready &&
            slot>=0.0 && slot<=9.0 &&
            exact_type>=0.0 && exact_type<=5.0 &&
            domain==1.0 &&
            prob_sum>0.99999 && prob_sum<1.00001 &&
            legal_count>0 &&
            cache_stable && duplicate_stable &&
            action_barrier &&
            first_inference==1.0;

        write_json(
            report,pass?"PASS":"FAIL",slot,exact_type,exact_amount,domain,
            prob_sum,legal_count,cache_stable&&duplicate_stable,action_barrier);

        std::cout<<"=== LT2 OPENHOLDEM SHADOW DLL MOCK HOST ===\n"
                 <<"VERDICT="<<(pass?"PASS":"FAIL")<<"\n"
                 <<"loaded="<<loaded<<" active="<<active<<" ready="<<ready
                 <<" failed="<<failed<<" domain="<<domain<<" slot="<<slot
                 <<" exact_type="<<exact_type<<" amount_to="<<exact_amount<<"\n"
                 <<"prob_sum="<<prob_sum<<" legal_count="<<legal_count
                 <<" cache_stable="<<(cache_stable&&duplicate_stable)
                 <<" action_barrier="<<action_barrier<<"\n"
                 <<"LT2_OPENHOLDEM_SHADOW_DLL_MOCK_HOST_COMPLETE\n";

        FreeLibrary(dll);
        return pass?0:2;
    } catch (const std::exception& e) {
        std::cerr<<"FATAL: "<<e.what()<<"\n";
        return 3;
    }
}
