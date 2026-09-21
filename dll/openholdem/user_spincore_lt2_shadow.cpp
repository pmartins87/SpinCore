#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>

#include "spincore/lt2_openholdem_runtime.hpp"
#include "spincore/lt2_openholdem_shadow_engine.hpp"

#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace fs=std::filesystem;

namespace {

using GetSymbolFn=double(*)(const char*);
using GetHandnumberFn=char*(*)();
using WriteLogFn=void(*)(char*,va_list);

HMODULE g_module=nullptr;
GetSymbolFn g_get_symbol=nullptr;
GetHandnumberFn g_get_handnumber=nullptr;
WriteLogFn g_write_log=nullptr;

std::mutex g_mutex;
fs::path g_dll_dir;
fs::path g_bundle_path;
fs::path g_log_path;

std::unique_ptr<spincore::lt2oh::NativeShadowDecisionEngine> g_engine;
std::optional<spincore::lt2oh::HandAnchor> g_anchor;
std::optional<spincore::lt2oh::ShadowDecision> g_last_decision;
bool g_bridge_failed=false;
std::string g_last_error;
std::uint64_t g_session_seed=0;
std::uint64_t g_hand_resets=0;
std::uint64_t g_heartbeats=0;
std::uint64_t g_myturns=0;

template<typename Fn>
Fn lookup_host(const char* name) {
    HMODULE host=GetModuleHandleW(nullptr);
    if (!host) return nullptr;
    return reinterpret_cast<Fn>(GetProcAddress(host,name));
}

void append_log(const std::string& line) noexcept {
    try {
        if (!g_log_path.empty()) {
            std::ofstream out(g_log_path,std::ios::app);
            if (out) out<<line<<"\n";
        }
        std::string debug="[SpinCore LT2 Shadow] "+line+"\n";
        OutputDebugStringA(debug.c_str());
    } catch (...) {
    }
}

std::string format(const char* fmt,...) {
    char buffer[4096]{};
    va_list args;
    va_start(args,fmt);
#if defined(_MSC_VER)
    vsnprintf_s(buffer,sizeof(buffer),_TRUNCATE,fmt,args);
#else
    std::vsnprintf(buffer,sizeof(buffer),fmt,args);
#endif
    va_end(args);
    return std::string(buffer);
}

bool host_api_ready() noexcept {
    return g_get_symbol!=nullptr && g_get_handnumber!=nullptr;
}

void initialize_host_api() noexcept {
    g_get_symbol=lookup_host<GetSymbolFn>("GetSymbol");
    g_get_handnumber=lookup_host<GetHandnumberFn>("GetHandnumber");
    g_write_log=lookup_host<WriteLogFn>("WriteLog");
}

void initialize_paths() noexcept {
    wchar_t path[MAX_PATH]{};
    if (g_module && GetModuleFileNameW(g_module,path,MAX_PATH)>0) {
        g_dll_dir=fs::path(path).parent_path();
        g_bundle_path=g_dll_dir/L"SpinCore_LT2_cpp_deployment_8100.bin";
        g_log_path=g_dll_dir/L"SpinCore_LT2_shadow.log";
    }
}

std::string sha256_file_hex(const fs::path& path) {
    BCRYPT_ALG_HANDLE alg=nullptr;
    BCRYPT_HASH_HANDLE hash=nullptr;
    DWORD object_len=0,cb=0,hash_len=0;
    std::vector<UCHAR> object;
    std::vector<UCHAR> digest;

    auto close_all=[&]() noexcept {
        if (hash) BCryptDestroyHash(hash);
        if (alg) BCryptCloseAlgorithmProvider(alg,0);
    };

    NTSTATUS status=BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0);
    if (status<0) {
        close_all();
        throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA256) failed");
    }
    status=BCryptGetProperty(
        alg,BCRYPT_OBJECT_LENGTH,
        reinterpret_cast<PUCHAR>(&object_len),sizeof(object_len),&cb,0);
    if (status<0 || object_len==0) {
        close_all();
        throw std::runtime_error("BCrypt SHA256 object-length query failed");
    }
    status=BCryptGetProperty(
        alg,BCRYPT_HASH_LENGTH,
        reinterpret_cast<PUCHAR>(&hash_len),sizeof(hash_len),&cb,0);
    if (status<0 || hash_len!=32U) {
        close_all();
        throw std::runtime_error("BCrypt SHA256 hash-length query failed");
    }
    object.resize(object_len);
    digest.resize(hash_len);
    status=BCryptCreateHash(
        alg,&hash,object.data(),static_cast<ULONG>(object.size()),
        nullptr,0,0);
    if (status<0) {
        close_all();
        throw std::runtime_error("BCryptCreateHash failed");
    }

    std::ifstream in(path,std::ios::binary);
    if (!in) {
        close_all();
        throw std::runtime_error("cannot open frozen deployment bundle");
    }
    std::array<char,65536> buffer{};
    while (in) {
        in.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));
        const auto n=in.gcount();
        if (n>0) {
            status=BCryptHashData(
                hash,
                reinterpret_cast<PUCHAR>(buffer.data()),
                static_cast<ULONG>(n),0);
            if (status<0) {
                close_all();
                throw std::runtime_error("BCryptHashData failed");
            }
        }
    }
    if (!in.eof()) {
        close_all();
        throw std::runtime_error("deployment bundle read failed");
    }

    status=BCryptFinishHash(hash,digest.data(),static_cast<ULONG>(digest.size()),0);
    if (status<0) {
        close_all();
        throw std::runtime_error("BCryptFinishHash failed");
    }
    close_all();

    std::ostringstream out;
    out<<std::hex<<std::setfill('0');
    for (const auto b:digest) out<<std::setw(2)<<static_cast<unsigned>(b);
    return out.str();
}

std::uint64_t secure_seed() {
    std::uint64_t value=0;
    const NTSTATUS status=BCryptGenRandom(
        nullptr,
        reinterpret_cast<PUCHAR>(&value),
        static_cast<ULONG>(sizeof(value)),
        BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status<0) throw std::runtime_error("BCryptGenRandom failed");
    return value;
}

void set_bridge_failure(const std::string& reason) noexcept {
    g_bridge_failed=true;
    g_last_error=reason;
    g_last_decision.reset();
    append_log("FAIL_CLOSED "+reason);
}

void clear_hand_state() noexcept {
    g_anchor.reset();
    g_last_decision.reset();
    g_bridge_failed=false;
    g_last_error.clear();
    if (g_engine) g_engine->reset();
}

bool ensure_engine_loaded() {
    if (g_engine) return true;
    if (!host_api_ready()) {
        set_bridge_failure("OpenHoldem host exports GetSymbol/GetHandnumber are unavailable");
        return false;
    }
    try {
        if (g_bundle_path.empty()) throw std::runtime_error("DLL bundle path not initialized");
        const auto actual=sha256_file_hex(g_bundle_path);
        if (actual!=spincore::lt2oh::kFrozenNativeBundleFileSha256) {
            throw std::runtime_error(
                "native deployment bundle SHA256 mismatch expected="+
                std::string(spincore::lt2oh::kFrozenNativeBundleFileSha256)+
                " actual="+actual);
        }
        auto bundle=spincore::lt2oh::load_frozen_deployment_bundle(g_bundle_path);
        g_session_seed=secure_seed();
        g_engine=std::make_unique<spincore::lt2oh::NativeShadowDecisionEngine>(
            std::move(bundle),g_session_seed);
        append_log(format(
            "MODEL_READY shadow_only=1 seed=%llu bundle=%s",
            static_cast<unsigned long long>(g_session_seed),
            g_bundle_path.string().c_str()));
        return true;
    } catch (const std::exception& e) {
        set_bridge_failure(std::string("model load failed: ")+e.what());
        return false;
    }
}

double required_symbol(const char* name) {
    if (!g_get_symbol) throw std::runtime_error("GetSymbol unavailable");
    const double value=g_get_symbol(name);
    if (!std::isfinite(value) || std::abs(value)>1.0e9) {
        throw std::runtime_error(std::string("invalid OpenHoldem symbol ")+name);
    }
    return value;
}

int required_int_symbol(const char* name) {
    const double value=required_symbol(name);
    const auto rounded=static_cast<long long>(std::llround(value));
    if (std::abs(value-static_cast<double>(rounded))>1.0e-6 ||
        rounded<static_cast<long long>(INT32_MIN) ||
        rounded>static_cast<long long>(INT32_MAX)) {
        throw std::runtime_error(std::string("non-integral OpenHoldem symbol ")+name);
    }
    return static_cast<int>(rounded);
}

int chair_chip_symbol(const char* stem,int chair) {
    char name[64]{};
#if defined(_MSC_VER)
    sprintf_s(name,"%s%d",stem,chair);
#else
    std::snprintf(name,sizeof(name),"%s%d",stem,chair);
#endif
    return required_int_symbol(name);
}

int card_from_oh_symbols(const char* rank_name,const char* suit_name) {
    const int rank=required_int_symbol(rank_name);
    const int suit=required_int_symbol(suit_name);
    return spincore::lt2oh::openholdem_card_id_from_rank_suit(rank,suit);
}

spincore::lt2oh::RawFrame capture_raw_frame() {
    if (!g_get_handnumber) throw std::runtime_error("GetHandnumber unavailable");
    char* hand=g_get_handnumber();
    if (!hand || *hand=='\0') throw std::runtime_error("empty OpenHoldem hand number");

    spincore::lt2oh::RawFrame frame{};
    frame.hand_id=hand;
    frame.user_chair=required_int_symbol("userchair");
    frame.dealer_chair=required_int_symbol("dealerchair");
    frame.betround=required_int_symbol("betround");
    frame.small_blind=required_int_symbol("sblind");
    frame.big_blind=required_int_symbol("bblind");
    frame.players_dealt_bits=static_cast<std::uint32_t>(required_int_symbol("playersdealtbits"));
    frame.players_playing_bits=static_cast<std::uint32_t>(required_int_symbol("playersplayingbits"));
    frame.players_allin_bits=static_cast<std::uint32_t>(required_int_symbol("playersallinbits"));
    frame.pot=required_int_symbol("pot");
    frame.common_cards_known=required_int_symbol("ncommoncardsknown");

    for (int chair=0;chair<10;++chair) {
        frame.balances[static_cast<std::size_t>(chair)]=chair_chip_symbol("balance",chair);
        frame.current_bets[static_cast<std::size_t>(chair)]=chair_chip_symbol("currentbet",chair);
    }

    frame.hero_cards[0]=card_from_oh_symbols("$$pr0","$$ps0");
    frame.hero_cards[1]=card_from_oh_symbols("$$pr1","$$ps1");

    frame.board_cards.fill(-1);
    for (int i=0;i<frame.common_cards_known && i<5;++i) {
        char rank_name[16]{};
        char suit_name[16]{};
#if defined(_MSC_VER)
        sprintf_s(rank_name,"$$cr%d",i);
        sprintf_s(suit_name,"$$cs%d",i);
#else
        std::snprintf(rank_name,sizeof(rank_name),"$$cr%d",i);
        std::snprintf(suit_name,sizeof(suit_name),"$$cs%d",i);
#endif
        frame.board_cards[static_cast<std::size_t>(i)]=
            card_from_oh_symbols(rank_name,suit_name);
    }
    return frame;
}

bool try_start_hand(const spincore::lt2oh::RawFrame& frame,bool myturn) {
    if (!g_engine && !ensure_engine_loaded()) return false;
    if (g_anchor) return true;

    if (frame.betround!=1 || frame.common_cards_known!=0) {
        if (myturn) set_bridge_failure("MyTurn reached before a valid preflop hand anchor");
        return false;
    }
    try {
        auto anchor=spincore::lt2oh::anchor_from_raw_frame(frame);
        auto observed=spincore::lt2oh::normalize_raw_frame(frame,anchor);
        const auto start=g_engine->start_hand(anchor,observed);
        if (start.kind!=spincore::lt2oh::SyncKind::Start) {
            if (myturn) set_bridge_failure(
                "failed to start canonical hand: "+start.reason);
            return false;
        }
        g_anchor=std::move(anchor);
        append_log(format(
            "HAND_START id=%s hero=%d domain=%s stacks=%d,%d,%d blinds=%d/%d",
            g_anchor->hand_id.c_str(),
            g_anchor->hero_logical_seat,
            g_anchor->scenario.state.game_is_hu?"HU":"3H",
            g_anchor->scenario.state.stacks[0],
            g_anchor->scenario.state.stacks[1],
            g_anchor->scenario.state.stacks[2],
            g_anchor->scenario.state.small_blind,
            g_anchor->scenario.state.big_blind));
        return true;
    } catch (const std::exception& e) {
        if (myturn) set_bridge_failure(std::string("hand anchor failed at MyTurn: ")+e.what());
        return false;
    }
}

void process_heartbeat_locked() {
    ++g_heartbeats;
    if (g_bridge_failed) return;
    if (!ensure_engine_loaded()) return;
    try {
        const auto frame=capture_raw_frame();
        if (!try_start_hand(frame,false)) return;
        const auto observed=spincore::lt2oh::normalize_raw_frame(frame,*g_anchor);
        const auto result=g_engine->heartbeat(*g_anchor,observed);
        if (result.kind==spincore::lt2oh::SyncKind::Failed) {
            set_bridge_failure("heartbeat reconciliation failed: "+result.reason);
            return;
        }
        if (result.kind==spincore::lt2oh::SyncKind::Action) {
            g_last_decision.reset();
            append_log(format(
                "SYNC generation=%llu transcript=%llu",
                static_cast<unsigned long long>(g_engine->tracker().generation()),
                static_cast<unsigned long long>(g_engine->tracker().transcript().size())));
        }
    } catch (const std::exception& e) {
        if (g_anchor) set_bridge_failure(std::string("heartbeat capture failed: ")+e.what());
    }
}

void process_myturn_locked() {
    ++g_myturns;
    if (g_bridge_failed) return;
    if (!ensure_engine_loaded()) return;
    try {
        const auto frame=capture_raw_frame();
        if (!try_start_hand(frame,true)) return;
        const auto observed=spincore::lt2oh::normalize_raw_frame(frame,*g_anchor);
        const auto decision=g_engine->on_my_turn(*g_anchor,observed);
        if (!decision.has_value()) {
            set_bridge_failure("MyTurn synchronization/inference returned no shadow decision");
            return;
        }
        g_last_decision=decision;
        append_log(format(
            "SHADOW_DECISION hand=%s gen=%llu idx=%llu domain=%d slot=%d exact_type=%d amount_to=%d u=%.17g",
            g_anchor->hand_id.c_str(),
            static_cast<unsigned long long>(decision->tracker_generation),
            static_cast<unsigned long long>(decision->decision_index),
            decision->domain,
            decision->action_slot,
            static_cast<int>(decision->exact.type),
            decision->exact.amount_to,
            decision->sample_u));
    } catch (const std::exception& e) {
        set_bridge_failure(std::string("MyTurn capture failed: ")+e.what());
    }
}

bool query_is(const char* q,const char* name) noexcept {
    return q && std::strcmp(q,name)==0;
}

double query_debug_locked(const char* q) noexcept {
    if (!q) return 0.0;

    // Hard safety barrier: the shadow DLL never authorizes a table action.
    for (const char* action_query:{
        "dll$fold","dll$check","dll$call","dll$rais",
        "dll$alli","dll$betsize","dll$deep_action"}) {
        if (query_is(q,action_query)) return 0.0;
    }

    if (query_is(q,"dll$spincore_shadow_only")) return 1.0;
    if (query_is(q,"dll$spincore_loaded")) return g_engine?1.0:0.0;
    if (query_is(q,"dll$spincore_hand_active")) return g_anchor?1.0:0.0;
    if (query_is(q,"dll$spincore_failed")) return g_bridge_failed?1.0:0.0;
    if (query_is(q,"dll$spincore_ready")) return g_last_decision.has_value()?1.0:0.0;
    if (query_is(q,"dll$spincore_hand_resets")) return static_cast<double>(g_hand_resets);
    if (query_is(q,"dll$spincore_heartbeats")) return static_cast<double>(g_heartbeats);
    if (query_is(q,"dll$spincore_myturns")) return static_cast<double>(g_myturns);
    if (query_is(q,"dll$spincore_inference_count")) {
        return g_engine?static_cast<double>(g_engine->inference_count()):0.0;
    }
    if (!g_last_decision.has_value()) return 0.0;

    const auto& d=*g_last_decision;
    if (query_is(q,"dll$spincore_generation")) return static_cast<double>(d.tracker_generation);
    if (query_is(q,"dll$spincore_decision_index")) return static_cast<double>(d.decision_index);
    if (query_is(q,"dll$spincore_domain")) return static_cast<double>(d.domain);
    if (query_is(q,"dll$spincore_action_slot")) return static_cast<double>(d.action_slot);
    if (query_is(q,"dll$spincore_exact_type")) return static_cast<double>(static_cast<int>(d.exact.type));
    if (query_is(q,"dll$spincore_exact_amount_to")) return static_cast<double>(d.exact.amount_to);
    if (query_is(q,"dll$spincore_sample_u")) return d.sample_u;

    for (int i=0;i<static_cast<int>(spincore::lt2::kActionCount);++i) {
        char prob[32]{},legal[32]{};
#if defined(_MSC_VER)
        sprintf_s(prob,"dll$spincore_p%d",i);
        sprintf_s(legal,"dll$spincore_l%d",i);
#else
        std::snprintf(prob,sizeof(prob),"dll$spincore_p%d",i);
        std::snprintf(legal,sizeof(legal),"dll$spincore_l%d",i);
#endif
        if (query_is(q,prob)) return static_cast<double>(d.probabilities[static_cast<std::size_t>(i)]);
        if (query_is(q,legal)) return static_cast<double>(d.legal_mask[static_cast<std::size_t>(i)]);
    }
    return 0.0;
}

void on_load() noexcept {
    initialize_paths();
    initialize_host_api();
    append_log(format(
        "DLL_LOAD shadow_only=1 host_api=%d dll=%s",
        host_api_ready()?1:0,
        g_dll_dir.string().c_str()));
}

void on_unload() noexcept {
    append_log("DLL_UNLOAD");
    g_last_decision.reset();
    g_anchor.reset();
    g_engine.reset();
}

} // namespace

extern "C" __declspec(dllexport) double __stdcall ProcessQuery(const char* pquery) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return query_debug_locked(pquery);
}

extern "C" __declspec(dllexport) void __stdcall DLLUpdateOnNewFormula() {
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_hand_state();
    (void)ensure_engine_loaded();
    append_log("CALLBACK NewFormula");
}

extern "C" __declspec(dllexport) void __stdcall DLLUpdateOnConnection() {
    std::lock_guard<std::mutex> lock(g_mutex);
    (void)ensure_engine_loaded();
    append_log("CALLBACK Connection");
}

extern "C" __declspec(dllexport) void __stdcall DLLUpdateOnHandreset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_hand_resets;
    clear_hand_state();
    append_log("CALLBACK Handreset");
}

extern "C" __declspec(dllexport) void __stdcall DLLUpdateOnNewRound() {
    std::lock_guard<std::mutex> lock(g_mutex);
    append_log("CALLBACK NewRound");
}

extern "C" __declspec(dllexport) void __stdcall DLLUpdateOnMyTurn() {
    std::lock_guard<std::mutex> lock(g_mutex);
    process_myturn_locked();
}

extern "C" __declspec(dllexport) void __stdcall DLLUpdateOnHeartbeat() {
    std::lock_guard<std::mutex> lock(g_mutex);
    process_heartbeat_locked();
}

BOOL APIENTRY DllMain(HMODULE module,DWORD reason,LPVOID) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            g_module=module;
            DisableThreadLibraryCalls(module);
            on_load();
            break;
        case DLL_PROCESS_DETACH:
            on_unload();
            break;
        default:
            break;
    }
    return TRUE;
}
