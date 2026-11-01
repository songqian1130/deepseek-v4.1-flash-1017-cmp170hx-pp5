// Pins the layer-split handoff order and the image identity a parked
// conversation must match before it is reused.
//
//   g++ -std=c++17 -Wall -Wextra -Iinclude -o ds41_group_pipe_test ds41_group_pipe_test.cpp
#include "ds41_group_pipe.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace v = strata::v41;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool same_launches(const std::vector<v::Launch>& got, const std::vector<v::Launch>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i].tick != want[i].tick || got[i].stage != want[i].stage || got[i].group != want[i].group)
            return false;
    return true;
}
v::Launch L(int tick, int stage, int group) { return v::Launch{ tick, stage, group }; }
}  // namespace

int main() {
    std::printf("ds41_group_pipe_test\n");

    check(v::auto_groups(0, 4) == 1, "auto: no layer split, one group");
    check(v::auto_groups(2, 2) == 2 && v::auto_groups(2, 3) == 3 && v::auto_groups(2, 4) == 2,
          "auto: three cards, the most groups that divide the slots");

    {
        v::GroupAsk a;
        a.batch = 2;
        a.later_stages = 2;
        a.batch_mtp = true;
        const v::GroupPlan r = v::resolve_groups(a);
        check(r.groups == 2 && r.from_auto && r.mtp_dropped && r.serial == v::GroupSerial::None,
              "three cards, two requests: pipeline, drafts off");
        a.mtp_serial = true;
        const v::GroupPlan s = v::resolve_groups(a);
        check(s.groups == 1 && s.serial == v::GroupSerial::BatchMtp && s.would_be == 2,
              "STRATA_BATCH_MTP_SERIAL keeps one group");
    }
    {
        v::GroupAsk a;
        a.batch = 1;
        a.later_stages = 2;
        const v::GroupPlan r = v::resolve_groups(a);
        check(r.groups == 1 && !r.from_auto, "one request: no pipeline");
    }

    check(v::split_batch_block(true, false) != nullptr, "split without a per-card slot still refuses");
    check(v::split_batch_block(true, true) == nullptr, "split with one slot image per card is allowed");
    check(v::split_batch_block(false, false) == nullptr, "no split: this gate stays quiet");

    {
        std::string err;
        v::GroupPipe pipe(3, 2, 1);
        check(pipe.drain(err), "two groups, three cards, drain");
        if (!err.empty()) std::printf("    %s\n", err.c_str());
        const std::vector<v::Launch> want = {
            L(1, 0, 0), L(2, 1, 0), L(2, 0, 1), L(3, 2, 0), L(3, 1, 1), L(4, 2, 1),
        };
        check(same_launches(pipe.launches(), want),
              "handoff: card 0 starts the next group while card 1 still has the previous one");
        check(pipe.conv(0).step == 1 && pipe.conv(1).step == 1, "each conversation produced its own token");
        check(pipe.conv(0).stage_tokens[0] == 1 && pipe.conv(0).stage_tokens[2] == 1 &&
                  pipe.conv(1).stage_tokens[0] == 1 && pipe.conv(1).stage_tokens[2] == 1,
              "every card kept that conversation's step");
        check(pipe.card(2).conv == 1 && pipe.card(2).image == pipe.conv(1).image,
              "the last card still holds the conversation that finished there");
        check(pipe.conv(0).image != pipe.conv(1).image, "the two conversations do not share an image identity");
    }

    {
        std::string err;
        v::GroupPipe pipe(3, 3, 1);
        check(pipe.drain(err), "one group per card, drain");
        const std::vector<v::Launch> want = {
            L(1, 0, 0),
            L(2, 1, 0), L(2, 0, 1),
            L(3, 2, 0), L(3, 1, 1), L(3, 0, 2),
            L(4, 2, 1), L(4, 1, 2),
            L(5, 2, 2),
        };
        check(same_launches(pipe.launches(), want), "three groups fill the three cards, then hand off in order");
        bool ahead = false;
        for (int g = 0; g < 3 && !ahead; ++g)
            for (int s = 1; s < 3; ++s)
                if (pipe.conv(g).stage_tokens[(size_t) s] > pipe.conv(g).stage_tokens[(size_t) s - 1]) ahead = true;
        check(!ahead && pipe.conv(0).step == 1 && pipe.conv(2).step == 1,
              "no card ran a step the previous card had not finished");
    }

    {
        std::string err;
        v::GroupPipe pipe(3, 1, 2);
        check(pipe.drain(err), "one request, two tokens, drain");
        bool overlap = false;
        for (size_t i = 1; i < pipe.launches().size(); ++i)
            if (pipe.launches()[i].tick == pipe.launches()[i - 1].tick &&
                pipe.launches()[i].group != pipe.launches()[i - 1].group)
                overlap = true;
        check(!overlap, "one request does not overlap a second conversation");
        check(pipe.conv(0).step == 2 && pipe.conv(0).stage_tokens[2] == 2,
              "the single conversation still finishes on the last card");
        int first_second = -1;
        for (const v::Launch& x : pipe.launches())
            if (x.stage == 0 && first_second < 0 && x.tick > 1) first_second = x.tick;
        int last_of_first = -1;
        for (const v::Launch& x : pipe.launches())
            if (x.stage == 2) { last_of_first = x.tick; break; }
        check(first_second > last_of_first, "the next token waits until the previous one leaves the last card");
    }

    {
        const std::vector<int> cached = { 10, 11, 12, 13 };
        const std::vector<int> prompt = { 10, 11, 12, 13, 14 };
        const v::ImageKey shot{ 1, 0xabc };
        check(v::reuse_tokens(cached, { shot }, prompt, { shot }) == 4,
              "same tokens and the same picture: reuse the prefix");
        check(v::reuse_tokens(cached, { shot }, prompt, { v::ImageKey{ 1, 0xdef } }) == 0,
              "same tokens, different picture: do not reuse");
        check(v::reuse_tokens(cached, { shot }, prompt, {}) == 0,
              "a pictured conversation is not reused by a text-only request");
        check(v::reuse_tokens(cached, {}, prompt, { v::ImageKey{ 1, 0xabc } }) == 0,
              "a text conversation is not reused when the request brings a picture inside the prefix");
        check(v::reuse_tokens(cached, {}, prompt, { v::ImageKey{ 4, 0xabc } }) == 4,
              "a picture that starts after the prefix does not block a text hit");
        check(v::reuse_tokens(cached, { shot }, { 10, 11, 99, 13, 14 }, { shot }) == 0,
              "a changed token refuses the hit even when the picture matches");
    }

    std::printf("%s\n", g_fail == 0 ? "ds41_group_pipe_test: all passed" : "ds41_group_pipe_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
