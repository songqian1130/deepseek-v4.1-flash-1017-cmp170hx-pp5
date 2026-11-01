// Layer-split batch groups for DeepSeek V4.1.
//
// The Qwen side of this tree already pipelines `--batch` (--batch-groups, see
// include/strata/program/batch_groups.hpp): each conversation keeps its own
// state on every card, and groups move one stage per turn.  V4.1 used to refuse
// that whenever a layer split was on, because one slot image was shared by
// every stage.  This scheduler is the V4.1 side of the same rule.
//
// Pure: no GPU, no environment.  A stage may start a group only after the
// previous card has finished that same step and handed it on.  One request
// (one group) still walks the cards in order; a second request overlaps.
// Drafts stay off once more than one group is in the pipe, same as Qwen.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::v41 {

struct ImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImageKey& o) const { return start == o.start && hash == o.hash; }
};

/// Why `auto` did not pipeline.  Mirrors strata::program::batch_groups::Serial.
enum class GroupSerial { None, BatchMtp, AdaptAsync };

struct GroupAsk {
    bool set = false;
    bool auto_given = false;
    int given = 1;
    int batch = 0;
    int later_stages = 0;   ///< stages after the first card (0: no layer split)
    bool batch_mtp = false;
    bool adapt_async = false;
    bool mtp_serial = false;
};

struct GroupPlan {
    int groups = 1;
    bool from_auto = false;
    GroupSerial serial = GroupSerial::None;
    int would_be = 1;
    bool mtp_dropped = false;   ///< the pipeline runs, so slot drafts stay off
};

inline int auto_groups(int later_stages, int batch) {
    int best = 1;
    if (later_stages <= 0) return best;
    for (int d = 2; d <= later_stages + 1 && d <= batch; ++d)
        if (batch % d == 0) best = d;
    return best;
}

inline GroupPlan resolve_groups(const GroupAsk& in) {
    GroupPlan r;
    r.groups = in.given;
    r.would_be = in.given;
    const bool defaulted = !in.set && in.later_stages > 0 && in.batch > 1;
    if (!in.auto_given && !defaulted) return r;
    r.from_auto = true;
    r.would_be = auto_groups(in.later_stages, in.batch);
    r.groups = r.would_be;
    if (!in.set && r.would_be > 1) {
        if (in.batch_mtp && in.mtp_serial) r.serial = GroupSerial::BatchMtp;
        else if (in.adapt_async) r.serial = GroupSerial::AdaptAsync;
        r.mtp_dropped = in.batch_mtp && r.serial == GroupSerial::None;
        if (r.serial != GroupSerial::None) r.groups = 1;
    }
    return r;
}

/// Empty when a layer split may keep one slot image per card.  Otherwise the
/// reason the serve loop still runs one request at a time.
inline const char* split_batch_block(bool layer_split, bool per_stage_slots) {
    if (!layer_split || per_stage_slots) return nullptr;
    return "a layer split runs one request at a time (the slots and the pool hold one stage's state)";
}

/// Longest cached prefix whose tokens and in-prefix images both match.
/// An image that starts at or after the prefix does not take part.  A cached
/// image the request does not repeat, or a different hash, refuses the hit.
inline int reuse_tokens(const std::vector<int>& cached_ids, const std::vector<ImageKey>& cached_imgs,
                        const std::vector<int>& prompt, const std::vector<ImageKey>& images) {
    const size_t n = cached_ids.size();
    if (n == 0 || n >= prompt.size() || !std::equal(cached_ids.begin(), cached_ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const ImageKey& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == cached_imgs.size() || !(cached_imgs[j++] == image)) return 0;
    }
    if (j != cached_imgs.size()) return 0;
    return (int) n;
}

struct Launch {
    int tick = 0;
    int stage = 0;
    int group = 0;
};

/// One conversation's state, copied onto every card.  `stage_tokens[s]` is how
/// many tokens card s has already computed for it.  A later card never gets
/// ahead of the card before it.
struct PipeConv {
    bool active = true;
    int step = 0;
    int target = 0;
    uint64_t image = 0;
    std::vector<int> stage_tokens;
};

struct CardHold {
    int conv = -1;         ///< whose state is sitting on this card
    uint64_t image = 0;    ///< that conversation's image identity
    int tokens = 0;        ///< tokens of that conversation computed here
};

/// The Qwen pump: poll every busy card, hand a finished group to the next
/// card, then start the longest-waiting group on a free later card and a new
/// group on card 0 (round-robin).  One group occupies one card at a time.
class GroupPipe {
public:
    GroupPipe(int n_stages, int n_groups, int steps, uint64_t image0 = 1)
        : n_stages_(std::max(1, n_stages)), n_groups_(std::max(1, n_groups)),
          stage_group_((size_t) std::max(1, n_stages), -1),
          conv_((size_t) std::max(1, n_groups)),
          cards_((size_t) std::max(1, n_stages)) {
        for (int g = 0; g < n_groups_; ++g) {
            conv_[(size_t) g].target = std::max(0, steps);
            conv_[(size_t) g].image = image0 + (uint64_t) g;
            conv_[(size_t) g].stage_tokens.assign((size_t) n_stages_, 0);
            conv_[(size_t) g].active = steps > 0;
        }
        pg_.assign((size_t) n_groups_, Group{});
    }

    int n_stages() const { return n_stages_; }
    int n_groups() const { return n_groups_; }
    const std::vector<Launch>& launches() const { return launches_; }
    const PipeConv& conv(int g) const { return conv_[(size_t) g]; }
    const CardHold& card(int s) const { return cards_[(size_t) s]; }
    bool busy() const {
        for (const Group& g : pg_) if (g.inflight) return true;
        for (const PipeConv& c : conv_) if (c.active) return true;
        return false;
    }

    /// One turn.  False when a later card would run a step the previous card
    /// has not handed off, or two groups would share a card.
    bool pump(bool may_start, std::string& err) {
        ++tick_;
        for (int k = 0; k < n_stages_; ++k) {
            const int gi = stage_group_[(size_t) k];
            if (gi < 0) continue;
            stage_group_[(size_t) k] = -1;
            Group& G = pg_[(size_t) gi];
            PipeConv& c = conv_[(size_t) gi];
            if (c.active) {
                c.stage_tokens[(size_t) k] += 1;
                if (k > 0 && c.stage_tokens[(size_t) k] > c.stage_tokens[(size_t) k - 1]) {
                    err = "stage ran ahead of the card before it";
                    return false;
                }
            }
            hold(k, gi);
            if (k + 1 < n_stages_) {
                G.stage = k + 1;
                G.since = tick_;
                continue;
            }
            if (c.active) {
                c.step += 1;
                if (c.step >= c.target) c.active = false;
            }
            G.inflight = false;
        }
        for (int k = n_stages_ - 1; k >= 0; --k) {
            if (stage_group_[(size_t) k] >= 0) continue;
            int pick = -1;
            if (k > 0) {
                for (int gi = 0; gi < n_groups_; ++gi) {
                    const Group& G = pg_[(size_t) gi];
                    if (G.inflight && G.stage == k && (pick < 0 || G.since < pg_[(size_t) pick].since)) pick = gi;
                }
                if (pick < 0) continue;
            } else {
                if (!may_start) continue;
                for (int j = 0; j < n_groups_ && pick < 0; ++j) {
                    const int gi = (rr_ + j) % n_groups_;
                    if (!pg_[(size_t) gi].inflight && conv_[(size_t) gi].active) pick = gi;
                }
                if (pick < 0) continue;
                rr_ = pick + 1;
                Group& G = pg_[(size_t) pick];
                G.inflight = true;
                G.stage = 0;
            }
            const PipeConv& c = conv_[(size_t) pick];
            if (k > 0 && c.stage_tokens[(size_t) k - 1] < c.stage_tokens[(size_t) k] + 1) {
                err = "handoff before the previous card finished this step";
                return false;
            }
            stage_group_[(size_t) k] = pick;
            launches_.push_back(Launch{ tick_, k, pick });
            hold(k, pick);
        }
        return true;
    }

    bool drain(std::string& err) {
        int idle = 0;
        const int guard = std::max(8, n_stages_ * n_groups_ * (conv_.empty() ? 1 : conv_[0].target + 2));
        while (busy()) {
            const size_t before = launches_.size();
            const int step0 = conv_[0].step;
            if (!pump(true, err)) return false;
            if (launches_.size() == before && conv_[0].step == step0) {
                if (++idle > 1) { err = "pump made no progress"; return false; }
            } else idle = 0;
            if ((int) launches_.size() > guard * 4) { err = "pump did not finish"; return false; }
        }
        return true;
    }

private:
    struct Group {
        bool inflight = false;
        int stage = 0;
        int since = 0;
    };
    void hold(int stage, int gi) {
        CardHold& h = cards_[(size_t) stage];
        h.conv = gi;
        h.image = conv_[(size_t) gi].image;
        h.tokens = conv_[(size_t) gi].stage_tokens[(size_t) stage];
    }
    int n_stages_ = 1;
    int n_groups_ = 1;
    int tick_ = 0;
    int rr_ = 0;
    std::vector<int> stage_group_;
    std::vector<Group> pg_;
    std::vector<PipeConv> conv_;
    std::vector<CardHold> cards_;
    std::vector<Launch> launches_;
};

}  // namespace strata::v41
