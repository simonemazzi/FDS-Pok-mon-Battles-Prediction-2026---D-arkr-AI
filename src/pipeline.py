import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from collections import defaultdict

import time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from scipy.optimize import minimize

#####################
#1. Config Vars
DATA_DIR = "/kaggle/input/competitions/fds-pokemon-battles-prediction-2026"
SEED = 42
#####################

#####################
#2. Data Pre-Processing

STATS= []
TYPES = []
STATUSES = []
EFFECTS = []
TOP_MOVES = []
TOP_ITEMS = []

BOOST_MULT = {s: ((2 + s) / 2 if s >= 0 else 2 / (2 - s)) for s in range(-6, 7)}

def collect_vocabulary(battles):
    """
    Collects the vocabulary of stats, types, statuses and effects from the battle data.
    """
    stats, types, statuses, effects = set(), set(), set(), set()
    for d in battles:
        for p in d["p1_team_details"]:
            types.update(t for t in p["types"] if t != "notype")
            stats.update(k[len("base_"):] for k in p if k.startswith("base_"))
        lead = d["p2_lead_details"]
        types.update(t for t in lead["types"] if t != "notype")
        for t in d["battle_timeline"]:
            for side in ("p1", "p2"):
                state = t[f"{side}_pokemon_state"]
                statuses.add(state["status"])
                effects.update(e for e in state["effects"] if e != "noeffect")
    return sorted(stats), sorted(types), sorted(statuses), sorted(effects)
def collect_top_items(battles, k = 8):
    """
    Returns the k most used items in the battle data.
    """
    counts: Counter = Counter()
    for d in battles:
        for p in d["p1_team_details"]:
            counts[p["item"]] += 1
    return [it for it, _ in counts.most_common(k)]
def collect_top_moves(battles, k = 50):
    """
    Returns the k most used moves in the battle data.
    """
    counts: Counter = Counter()
    for d in battles:
        for t in d["battle_timeline"]:
            for side in ("p1", "p2"):
                m = t[f"{side}_move_details"]
                if m is not None:
                    counts[m["name"]] += 1
    return [mv for mv, _ in counts.most_common(k)]
def set_vocabulary(battles):
    """
    Populates the global vocabulary lists (STATS, TYPES, STATUSES, EFFECTS, TOP_MOVES, TOP_ITEMS) based on the provided battle data.
    """
    stats, types, statuses, effects = collect_vocabulary(battles)
    STATS[:] = stats
    TYPES[:] = types
    STATUSES[:] = statuses
    EFFECTS[:] = effects
    TOP_MOVES[:] = collect_top_moves(battles)
    TOP_ITEMS[:] = collect_top_items(battles)
def load_jsonl(name):
    with open(f"{DATA_DIR}/{name}") as f:
        return [json.loads(line) for line in f]
def load_type_chart():
    df = pd.read_csv(f"{DATA_DIR}/type_chart_gen2.csv")
    chart = {(r.attacking_type, r.defending_type): float(r.multiplier)
             for r in df.itertuples()}
    return chart
def build_pokemon_table(include_test = True):
    """
    Builds a table from the training and test data, containing types and base stats for each pokemon.
    """
    train = load_jsonl("train.jsonl")
    test = load_jsonl("test.jsonl")
    set_vocabulary(train + test)
    table: dict[str, dict[str, Any]] = {}
    source_data = train + test if include_test else train
    for d in source_data:
        for p in d["p1_team_details"]:
            table[p["name"]] = {
                "types": list(p["types"]),
                **{s: float(p[f"base_{s}"]) for s in STATS},
            }
        p = d["p2_lead_details"]
        table[p["name"]] = {
            "types": list(p["types"]),
            **{s: float(p[f"base_{s}"]) for s in STATS},
        }
    med = {s: float(np.median([v[s] for v in table.values()])) for s in STATS}
    for d in source_data:
        for t in d["battle_timeline"]:
            n = t["p2_pokemon_state"]["name"]
            if n not in table:
                table[n] = {"types": ["notype", "notype"], **med}
    return table
def defensive_mult(chart, def_types, atk_type):
    """
    Calculates the defensive multiplier for a given attack type against a set of defending types using the provided type chart.
    """
    m = 1.0
    for t in def_types:
        if t == "notype":
            continue
        m *= chart.get((atk_type, t), 1.0)
    return m
def best_stab_eff(chart, atk_types, def_types):
    """
    Calculates the best effectiveness multiplier for a set of attacking types against a set of defending types using the provided type chart.
    PSA: "STAB" stands for "Same Type Attack Bonus" and is a multiplier applied to moves that match the user's type.
    """
    vals = [defensive_mult(chart, def_types, t) for t in atk_types if t != "notype"]
    return max(vals) if vals else 1.0
def move_expected_power(chart, move, atk_types, def_types):
    """
    Calculates the expected power of a move given the attacking and defending types, using the provided type chart.
    """
    if move["category"] == "status" or move["base_power"] == 0:
        return 1.0, 1.0, 0.0
    eff = defensive_mult(chart, def_types, move["type"])
    stab = 1.5 if move["type"] in atk_types else 1.0
    return eff, stab, move["base_power"] * eff * stab * move["accuracy"]
def estimate_damage_frac(level, power, atk_stat, def_stat, max_hp, eff, stab, accuracy, item_mult = 1.0):
    """
    Estimates the fraction of damage dealt by a move given the attacker's level, move power, attack and defense stats, maximum HP of the defender, effectiveness multiplier, STAB multiplier, accuracy, and item multiplier.
    """
    if power <= 0:
        return 0.0
    lvl = max(1.0, float(level))
    base = (((2 * lvl / 5 + 2) * power * (atk_stat * item_mult) / max(1.0, def_stat)) / 50.0) + 2.0
    return base * eff * stab * accuracy / max(1.0, max_hp)
def team_to_pokemon_types_and_stats(p: dict) -> dict:
    """
    Converts a team dictionary to a dictionary containing the types and base stats of the pokemon in the team.
    """
    return {
        "types": list(p["types"]),
        **{s: float(p[f"base_{s}"]) for s in STATS},
    }
def stat_of(pokemons, name):
    return float(pokemons.get(name, 0.0))
def max_hp_of(pokemons, level):
    """
    Returns the maximum HP of a pokemon given its base stats and level.
    """
    return (2.0 * pokemons["hp"] + 31.0) * level / 100.0 + level + 10.0
def stat_value(pokemons, stat, level):
    """
    Returns the value of a specific stat for a pokemon given its base stats and level.
    """
    if stat == "hp":
        return max_hp_of(pokemons, level)
    return (2.0 * pokemons[stat] + 31.0) * level / 100.0 + 5.0
def pokemons_features(prefix, sp):
    """
    Returns a dictionary of features for a given pokemon, including its stats, types, base stat total (bst, yes I know that it is easily confused with boost, me bad), offensiveness, bulkiness.
    """
    f = {}
    for s in STATS:
        f[f"{prefix}_{s}"] = stat_of(sp, s)
    f[f"{prefix}_bst"] = sum(stat_of(sp, s) for s in STATS)
    f[f"{prefix}_off"] = max(sp["atk"], sp["spa"])
    f[f"{prefix}_bulk"] = sp["hp"] * (sp["def"] + sp["spd"]) / 2.0
    for t in TYPES:
        f[f"{prefix}_type_{t}"] = 1.0 if t in sp["types"] else 0.0
    return f
def move_damage_frac(chart, pokemons, move, att_name, att_lvl, att_boosts, att_status, def_name, def_lvl, def_boosts):
    """
    Returns the estimated fraction of damage dealt by a move given the attacking and defending pokemons, levels, boosts, and status, using the provided type chart.
    """
    if move is None or move["category"] == "status" or move["base_power"] <= 0:
        return 0.0
    a = pokemons.get(att_name)
    d = pokemons.get(def_name)
    if a is None or d is None:
        return 0.0
    eff = defensive_mult(chart, d["types"], move["type"])
    if move["category"] == "physical":
        atk = stat_value(a, "atk", att_lvl) * BOOST_MULT[int(att_boosts["atk"])]
        if att_status == "brn":
            atk *= 0.5
        dfn = stat_value(d, "def", def_lvl) * BOOST_MULT[int(def_boosts["def"])]
    else:
        atk = stat_value(a, "spa", att_lvl) * BOOST_MULT[int(att_boosts["spa"])]
        dfn = stat_value(d, "spd", def_lvl) * BOOST_MULT[int(def_boosts["spd"])]
    stab = 1.5 if move["type"] in a["types"] else 1.0
    return estimate_damage_frac(att_lvl, move["base_power"], atk, dfn, max_hp_of(d, def_lvl), eff, stab, move["accuracy"])
def ko_candidates(tl, side, pokemons, levels, chart):
    """
    Returns a list of candidate pokemon that could be knocked out by the opponent's last move, based on the battle timeline, side, pokemons, levels, and type chart.
    """
    opp = "p2" if side == "p1" else "p1"
    last_idx = {}
    hp_at_last = {}
    fnt = set()
    for i, t in enumerate(tl):
        st = t[f"{side}_pokemon_state"]
        n = st["name"]
        last_idx[n] = i
        hp_at_last[n] = float(st["hp_pct"])
        if st["status"] == "fnt":
            fnt.add(n)
    cands = []
    for n, li in last_idx.items():
        if li == len(tl) - 1 or n in fnt:
            continue
        h = hp_at_last.get(n, 1.0)
        t = tl[li]
        md = t[f"{opp}_move_details"]
        ost = t[f"{opp}_pokemon_state"]
        est = move_damage_frac(chart, pokemons, md, ost["name"], levels.get(ost["name"], 100.0), ost["boosts"], ost["status"], n, levels.get(n, 100.0), t[f"{side}_pokemon_state"]["boosts"])
        if est >= 0.8 * h or h < 0.1:
            cands.append((n, h, est))
    return cands

# One-time data setup: load the data and build the shared tables once.
train = load_jsonl("train.jsonl")
test = load_jsonl("test.jsonl")
chart = load_type_chart()
pokemons = build_pokemon_table()
#####################

#####################
#3. Hidden P2 KO

KO_MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    random_state=SEED,
)

def event_features(turn, pokemon, hp_before, opponent_move, opponent_state, levels, turn_index, appearance_turns, active_turn_counts, last_seen_turns, type_chart, pokemons):
    """
    Returns a dictionary of features for a given event in the battle timeline, including the pokemon's HP fraction, turn number, active turn counts, last seen turns, estimated damage from the opponent's move, and various other features related to the pokemon and opponent's state.
    """
    features = {
        "hp_fraction": hp_before,
        "hp_fraction_squared": hp_before * hp_before,
        "hp_below_10pct": float(hp_before < 0.1),
        "hp_below_20pct": float(hp_before < 0.2),
        "hp_below_30pct": float(hp_before < 0.3),
        "hp_below_50pct": float(hp_before < 0.5),
        "turn": float(turn["turn"]),
        "active_turns_before": float(active_turn_counts.get(pokemon, 0)),
        "turns_since_last_seen": float(
            turn["turn"] - last_seen_turns.get(pokemon, 0)
        ),
    }

    estimated_damage = 0.0
    base_power = 0.0
    type_effectiveness = 0.0
    accuracy = 0.0
    priority = 0.0
    move_category = 0.0

    if opponent_move is not None:
        base_power = float(opponent_move["base_power"])
        accuracy = float(opponent_move["accuracy"])
        priority = float(opponent_move["priority"])
        move_category = (1.0 if opponent_move["category"] == "physical" else 2.0 if opponent_move["category"] == "special" else 0.0)
        if opponent_move["category"] != "status":
            estimated_damage = move_damage_frac(
                type_chart, pokemons, opponent_move,
                opponent_state["name"],
                levels.get(opponent_state["name"], 100.0),
                opponent_state["boosts"],
                opponent_state["status"],
                pokemon,
                levels.get(pokemon, 100.0),
                {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
            )
            if pokemon in pokemons and opponent_state["name"] in pokemons:
                type_effectiveness = defensive_mult(
                    type_chart,
                    pokemons[pokemon]["types"],
                    opponent_move["type"],
                )

    features.update({
        "estimated_damage": estimated_damage,
        "estimated_damage_over_hp": estimated_damage / max(hp_before, 0.02),
        "base_power": base_power,
        "type_effectiveness": type_effectiveness,
        "accuracy": accuracy,
        "priority": priority,
        "move_category": move_category,
        "no_move": float(opponent_move is None),
        "opponent_hp_fraction": float(opponent_state["hp_pct"]),
        "opponent_boost_sum": float(sum(opponent_state["boosts"].values())),
        "opponent_status_id": float(STATUSES.index(opponent_state["status"])),
    })

    features["reappears_later"] = float(any(later_turn > turn_index for later_turn in appearance_turns.get(pokemon, [])))

    if pokemon in pokemons:
        for stat_name in STATS:
            features[f"pokemon_{stat_name}"] = float(
                pokemons[pokemon][stat_name]
            )
        features["pokemon_bst"] = float(
            sum(pokemons[pokemon][stat_name] for stat_name in STATS)
        )
        features["pokemon_offense"] = max(
            pokemons[pokemon]["atk"],
            pokemons[pokemon]["spa"],
        )

    return features
def battle_events(battle, side, type_chart, pokemons):
    """
    Switch/faint events for one side of one battle.
    Each event is (battle_id, features, fainted_label, pokemon_name):

    With the timeline, the function iterates through the turns, keeping track of the previous pokemon and its HP. When a faint is observed, it creates an event with label 1.0. When a switch occurs (the active pokemon changes), it creates an event with label 0.0 for the previous pokemon.
    Returns the list of events.
    """

    timeline = battle["battle_timeline"]
    opponent_side = "p2" if side == "p1" else "p1"
    known_team = ( battle["p1_team_details"] if side == "p1" else [battle["p2_lead_details"]] )
    levels = {member["name"]: float(member["level"]) for member in known_team}

    appearance_turns = {}
    for turn_index, turn in enumerate(timeline):
        pokemon = turn[f"{side}_pokemon_state"]["name"]
        appearance_turns.setdefault(pokemon, []).append(turn_index)

    events = [] #List of events.
    previous_pokemon = None #With this we can catch switches.
    previous_hp = 1.0
    active_turn_counts = {}
    last_seen_turns = {}

    for turn_index, turn in enumerate(timeline):
        state = turn[f"{side}_pokemon_state"]
        pokemon = state["name"]
        opponent_state = turn[f"{opponent_side}_pokemon_state"]
        opponent_move = turn[f"{opponent_side}_move_details"]

        #Catch a fainted pokemon
        if state["status"] == "fnt":
            hp_before = previous_hp if previous_pokemon == pokemon else 1.0
            events.append((
                battle["battle_id"],
                event_features(
                    turn, pokemon, hp_before, opponent_move, opponent_state,
                    levels, turn_index, appearance_turns, active_turn_counts,
                    last_seen_turns, type_chart, pokemons,
                ),
                1.0,
                pokemon,
            ))
        #Catch a pokemon switch
        if previous_pokemon is not None and pokemon != previous_pokemon:
            events.append((
                battle["battle_id"],
                event_features(
                    turn, previous_pokemon, previous_hp,
                    opponent_move, opponent_state,
                    levels, turn_index, appearance_turns, active_turn_counts,
                    last_seen_turns, type_chart, pokemons,
                ),
                0.0,
                previous_pokemon,
            ))

        previous_pokemon = pokemon
        previous_hp = state["hp_pct"]
        active_turn_counts[pokemon] = active_turn_counts.get(pokemon, 0) + 1
        last_seen_turns[pokemon] = turn["turn"]

    return events
def build_events(battles, type_chart, pokemons):
    p1_events = []
    p2_events = []
    for battle in battles:
        p1_events += battle_events(battle, "p1", type_chart, pokemons)
        p2_events += battle_events(battle, "p2", type_chart, pokemons)
    return p1_events, p2_events
def fit_faint_model(train_p1):
    """
    Train a HistGradientBoostingClassifier on the P1 events to predict fainted labels. Returns the trained model and the list of feature columns.
    """
    features = pd.DataFrame([event[1] for event in train_p1])
    labels = np.array([event[2] for event in train_p1])
    model = HistGradientBoostingClassifier(**KO_MODEL_PARAMS)
    model.fit(features, labels)
    return model, list(features.columns)
def print_event_metrics(labels, probabilities, candidate_mask, threshold=0.5, title=""):
    """
    Log function, prints the metrics of the model.
    """
    print(f"\n### {title or 'Metrics'}\n")
    print(f"- **Events:** {len(labels):,} | **Positives:** {labels.mean():.1%} | **Candidates:** {candidate_mask.mean():.1%}\n")

    headers = ["Subset", "N", "Pos%", "Acc", "Prec", "Rec", "F1", "AUC", "AP", "TN", "FP", "FN", "TP"]
    alignments = [":---"] + [":---:"] * (len(headers) - 1)

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(alignments) + " |")

    subsets = (
        ("All events (P1)", np.ones(len(labels), dtype=bool)),
        ("Candidates (reappears=0)", candidate_mask),
    )

    for name, mask in subsets:
        y_true = labels[mask]
        y_score = probabilities[mask]
        y_pred = (y_score >= threshold).astype(int)

        if len(np.unique(y_true)) > 1:
            auc = f"{roc_auc_score(y_true, y_score):.4f}"
            ap = f"{average_precision_score(y_true, y_score):.4f}"
        else:
            auc = ap = "N/A"

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        row = [
            name,
            f"{len(y_true):,}",
            f"{y_true.mean():.1%}",
            f"{accuracy_score(y_true, y_pred):.4f}",
            f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
            f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
            f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
            auc,
            ap,
            f"{tn:,}",
            f"{fp:,}",
            f"{fn:,}",
            f"{tp:,}",
        ]
        print("| " + " | ".join(row) + " |")
def evaluate_holdout(model, feature_columns, events, threshold=0.5, title="holdout"):
    """
    Evaluate an already-trained model on labeled events (e.g. test P1). Basically call print_event_metrics with the input arguments.
    """
    features = pd.DataFrame(
        [event[1] for event in events]
    ).reindex(columns=feature_columns, fill_value=0)
    labels = np.array([event[2] for event in events])
    probabilities = model.predict_proba(features)[:, 1]
    candidate_mask = features["reappears_later"].to_numpy() == 0

    print_event_metrics(labels, probabilities, candidate_mask, threshold, title)
def evaluate_faint_model(train_p1, n_splits=5, threshold=0.5):
    """
    GroupKFold by battle on the P1 events (optional internal validation). Takes the values of the features and labels from the events and performs cross-validation using GroupKFold to ensure that events from the same battle are not split across training and validation sets. It prints the AUC and Average Precision for each fold, as well as overall metrics for all events and candidate events (those that do not reappear later). Returns the out-of-fold predictions for further analysis.
    """
    features = pd.DataFrame([event[1] for event in train_p1])
    labels = np.array([event[2] for event in train_p1])
    groups = np.array([event[0] for event in train_p1])
    candidate_mask = features["reappears_later"].to_numpy() == 0

    oof = np.zeros(len(labels))
    folds = GroupKFold(n_splits=n_splits).split(features, labels, groups)

    print(f"CV GroupKFold on {len(np.unique(groups))} battles:")
    for fold, (train_idx, valid_idx) in enumerate(folds, 1):
        model = HistGradientBoostingClassifier(**KO_MODEL_PARAMS)
        model.fit(features.iloc[train_idx], labels[train_idx])
        oof[valid_idx] = model.predict_proba(features.iloc[valid_idx])[:, 1]

        mask = candidate_mask[valid_idx]
        y_valid = labels[valid_idx][mask]
        p_valid = oof[valid_idx][mask]
        if len(np.unique(y_valid)) > 1:
            print(
                f"  fold {fold}: AUC={roc_auc_score(y_valid, p_valid):.4f} "
                f"AP={average_precision_score(y_valid, p_valid):.4f}"
            )
        else:
            print(f"  fold {fold}: one single class in validation set")

    print_event_metrics(
        labels, oof, candidate_mask, threshold, "CV train P1 (OOF)"
    )
    return oof
def aggregate_hidden_faints(model, feature_columns, eval_p2, n_battles):
    """
    Apply the trained model to P2 switch events and aggregate per battle.

    """
    features = pd.DataFrame(
        [event[1] for event in eval_p2]
    ).reindex(columns=feature_columns, fill_value=0)
    faint_probability = model.predict_proba(features)[:, 1]

    candidates = pd.DataFrame({
        "battle_id": [event[0] for event in eval_p2],
        "faint_probability": faint_probability,
        "observed_faint": [event[2] for event in eval_p2],
        "reappears_later": [event[1]["reappears_later"] for event in eval_p2],
        "hp_fraction": [event[1]["hp_fraction"] for event in eval_p2],
    })

    # Only switch events whose Pokemon never reappears can be hidden faints.
    candidates = candidates[
        (candidates.observed_faint == 0)
        & (candidates.reappears_later == 0)
    ]
    candidates["expected_lost_hp"] = (
        candidates.faint_probability * candidates.hp_fraction
    )

    aggregated = candidates.groupby("battle_id").agg(
        ko_hidden_prob=("faint_probability", "sum"),
        ko_hidden_hp=("expected_lost_hp", "sum"),
        ko_n_cand=("faint_probability", "size"),
        ko_max_prob=("faint_probability", "max"),
        ko_hidden_prob_gt05=("faint_probability", lambda s: float((s > 0.5).sum())),
        ko_hidden_prob_gt08=("faint_probability", lambda s: float((s > 0.8).sum())),
    )
    return aggregated.reindex(range(n_battles)).fillna(0.0)
def fit_and_aggregate(train_p1, eval_p2, n_battles):
    """
    Compatibility wrapper: fit once, then aggregate.
    """
    model, feature_columns = fit_faint_model(train_p1)
    return aggregate_hidden_faints(model, feature_columns, eval_p2, n_battles)

train = load_jsonl("train.jsonl")
test = load_jsonl("test.jsonl")
chart = load_type_chart()
pokemons = build_pokemon_table()

tr_p1, tr_p2 = build_events(train, chart, pokemons)
te_p1, te_p2 = build_events(test, chart, pokemons)
print(f"Train: {len(tr_p1):,} (P1) / {len(tr_p2):,} (P2)")
print(f"Test: {len(te_p1):,} (P1) / {len(te_p2):,} (P2)")

ko_oof = evaluate_faint_model(tr_p1, n_splits=5)

# Train on P1 and evaluate on test P1 (holdout) to see if the model generalizes well.
ko_model, ko_columns = fit_faint_model(tr_p1)
evaluate_holdout(ko_model, ko_columns, te_p1, title="test P1 (holdout)")

# Now we can use the trained model to aggregate hidden faints in P2 events.
agg_tr = aggregate_hidden_faints(ko_model, ko_columns, tr_p2, len(train))
agg_te = aggregate_hidden_faints(ko_model, ko_columns, te_p2, len(test))
print("aggregates: ", agg_tr.shape, agg_te.shape)


# ko_model_final, ko_columns = fit_faint_model(tr_p1 + te_p1)
# agg_tr = aggregate_hidden_faints(ko_model_final, ko_columns, tr_p2, len(train))
# agg_te = aggregate_hidden_faints(ko_model_final, ko_columns, te_p2, len(test))
#####################

#####################
#4. More Features! (Move Priors and Game State)
LEVEL = 100.0
EPS = 1e-4

def pokemons_move_priors(battles, alpha: float = 0.05):
    """
    Computes the prior probabilities of moves for each pokemon based on the battle data, using Laplace smoothing with parameter alpha.

    Recall from the statistics course: P(move | pokemon) = (k + alpha) / (N + alpha * V), where k is the count of the move for the pokemon, N is the total count of moves for the pokemon, and V is the vocabulary size (number of unique moves).
    In the end we have for each pokemon a list of (move, probability) pairs sorted by probability, and a global prior for pokemons without observations.

    """
    counts = defaultdict(Counter)
    details = {}
    global_counts = Counter()
    for d in battles:
        for t in d["battle_timeline"]:
            for side in ("p1", "p2"):
                m = t[f"{side}_move_details"]
                if m is None:
                    continue
                sp = t[f"{side}_pokemon_state"]["name"]
                counts[sp][m["name"]] += 1
                global_counts[m["name"]] += 1
                details[m["name"]] = m
    vocab = len(global_counts)
    priors = {}
    for sp, c in counts.items():
        tot = sum(c.values())
        items = []
        for mv, k in c.items():
            p = (k + alpha) / (tot + alpha * vocab)
            items.append((details[mv], p))
        items.sort(key=lambda x: -x[1])
        priors[sp] = items

    # global prior for pokemons without observations (pokemon not in priors)
    gtot = sum(global_counts.values())
    gitems = [(details[mv], (k + alpha) / (gtot + alpha * vocab)) for mv, k in global_counts.items()]
    gitems.sort(key=lambda x: -x[1])
    return priors, gitems
def median_pokemons(pokemons):
    """
    Computes the median base stats for a given pokemon dictionary, returning a dictionary with "notype" for types and the median values for each stat in STATS.
    """
    return {"types": ["notype", "notype"], **{s: float(np.median([v[s] for v in pokemons.values()])) for s in STATS}}
def attack_stats(sp: dict, move: dict, boosts: dict, status: str) -> tuple[float, float]:
    """
    Computes the attack stat for a given pokemon, move, boosts, and status.
    This is the formula used (written in LATEX):
    \\[
    \text{stat} = \frac{(2 \\cdot \text{base} + 31) \\cdot L}{100} + 5
    \\]
    """
    phys = move["category"] == "physical"
    if phys:
        atk = stat_value(sp, "atk", LEVEL) * BOOST_MULT[int(boosts.get("atk", 0))]
        if status == "brn":
            atk *= 0.5
    else:
        atk = stat_value(sp, "spa", LEVEL) * BOOST_MULT[int(boosts.get("spa", 0))]
    return atk, 1.0
def move_damage(chart, att_sp, def_sp, move, att_boosts, att_status, def_boosts):
    """
    Estimates the damage dealt by a move from an attacking pokemon to a defending pokemon, given their pokemon, boosts, and status.
    """
    if move["category"] == "status" or move["base_power"] <= 0:
        return 0.0
    phys = move["category"] == "physical"
    eff = defensive_mult(chart, def_sp["types"], move["type"])
    atk, item_mult = attack_stats(att_sp, move, att_boosts, att_status)
    dfn = stat_value(def_sp, "def" if phys else "spd", LEVEL) * BOOST_MULT[int(def_boosts.get("def" if phys else "spd", 0))]
    stab = 1.5 if move["type"] in att_sp["types"] else 1.0
    dmg = estimate_damage_frac(LEVEL, move["base_power"], atk, dfn, max_hp_of(def_sp, LEVEL), eff, stab, move["accuracy"], item_mult)
    return min(1.5, max(0.0, dmg))

ZERO_BOOSTS = {k: 0 for k in ("atk", "def", "spa", "spd", "spe")}

def expected_damage(chart, att_sp, def_sp, priors, att_boosts=None, att_status="nostatus", def_boosts=None, def_status="nostatus", top_n=14):
    """
    Computes the expected damage dealt by an attacking pokemon to a defending pokemon, given their pokemon, boosts, status, and item, using the provided type chart and move priors.
    """
    if att_boosts is None:
        att_boosts = ZERO_BOOSTS
    if def_boosts is None:
        def_boosts = ZERO_BOOSTS
    tot = 0.0
    best = 0.0
    cover = 0.0
    status_move = 0.0
    for move, p in priors[:top_n]:
        if move["category"] == "status":
            status_move += p
            continue
        d = move_damage(chart, att_sp, def_sp, move, att_boosts, att_status, def_boosts)
        tot += p * d
        if d > best:
            best = d
        if d > 0:
            eff = defensive_mult(chart, def_sp["types"], move["type"])
            if eff > 1:
                cover += p
    return tot, best, cover, status_move
def battle_state(d, pokemons, chart=None):
    """
    Extracts the battle state from a given battle dictionary and pokemon dictionary, returning a dictionary containing the battle timeline, player 1 and player 2 information, HP fractions, status conditions, last seen turns, and differences in HP fractions between players.
    Over all the timeline, we update the HP and status of the active pokemon for both players, and compute the difference in total HP fractions between players.
    Returns the battle state as a dictionary.
    """
    tl = d["battle_timeline"]
    p1team = d["p1_team_details"]
    lead2 = d["p2_lead_details"]

    p1_info = {}
    for p in p1team:
        sp = {**pokemons.get(p["name"], {}), "_name": p["name"]}
        sp["types"] = list(p["types"])
        for s in STATS:
            sp[s] = float(p[f"base_{s}"])
        p1_info[p["name"]] = dict(sp=sp, item=p["item"], level=float(p["level"]))

    p2_info = {}
    sp2 = pokemons.get(lead2["name"], median_pokemons(pokemons))
    p2_info[lead2["name"]] = dict(sp={**sp2, "_name": lead2["name"], "types": list(lead2["types"])}, item=None, level=float(lead2["level"]))

    hp1 = {n: 1.0 for n in p1_info}
    hp2 = {n: 1.0 for n in p2_info}
    st1 = {n: "nostatus" for n in p1_info}
    st2 = {n: "nostatus" for n in p2_info}
    last1 = {n: 0 for n in p1_info}
    last2 = {n: 0 for n in p2_info}

    diff = []
    active_diff = []
    for i, t in enumerate(tl):
        s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
        n1, n2 = s1["name"], s2["name"]
        if n1 not in p1_info:
            sp = {**pokemons.get(n1, median_pokemons(pokemons)), "_name": n1}
            p1_info[n1] = dict(sp=sp, item=None, level=100.0)
            hp1[n1] = 1.0
            st1[n1] = "nostatus"
            last1[n1] = 0
        if n2 not in p2_info:
            sp = {**pokemons.get(n2, median_pokemons(pokemons)), "_name": n2}
            p2_info[n2] = dict(sp=sp, item=None, level=100.0)
            hp2[n2] = 1.0
            st2[n2] = "nostatus"
            last2[n2] = 0
        dead1 = s1["hp_pct"] == 0 or s1["status"] == "fnt"
        dead2 = s2["hp_pct"] == 0 or s2["status"] == "fnt"
        hp1[n1] = 0.0 if dead1 else float(s1["hp_pct"])
        hp2[n2] = 0.0 if dead2 else float(s2["hp_pct"])
        st1[n1] = s1["status"]
        st2[n2] = s2["status"]
        last1[n1] = i
        last2[n2] = i
        tot1 = sum(hp1.values())
        tot2 = sum(hp2.values()) + (6 - len(hp2))
        diff.append(tot1 - tot2)
        active_diff.append(s1["hp_pct"] - s2["hp_pct"])

    return dict(tl=tl, p1_info=p1_info, p2_info=p2_info, hp1=hp1, hp2=hp2,
                st1=st1, st2=st2, last1=last1, last2=last2,
                diff=np.array(diff), active_diff=np.array(active_diff),
                n=len(tl))
#####################

#####################
#5. Moves Effects

NO_STATUS = ("nostatus", "fnt")
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe")
PROFILE_KEYS = ("boost", "heal", "self_ko", "opp_switch", "entry_dmg", "damage", "priority")

def real_statuses():
    """
    Statuses that actually affect a pokemon, except 'non status' and 'fnt'.
    """
    return [s for s in STATUSES if s not in NO_STATUS]
def build_profile(battles):
    """
    Builds a profile of move usage and effects from a list of battles.
    Returns a dictionary mapping move names to their usage statistics and average effects on
    boosts, healing, self-KOs, opponent switches, entry damage, damage dealt, priority moves, and status inflictions.
    """
    uses: Counter = Counter()
    sums: dict[str, defaultdict] = {k: defaultdict(float) for k in PROFILE_KEYS}
    status_sums: dict[str, defaultdict] = {s: defaultdict(float) for s in real_statuses()}

    for d in battles: #Discover all battles and their moves, and compute the effects of each move on the battle state.
        tl = d["battle_timeline"]
        for i, t in enumerate(tl): # For all turns in the battle timeline, we check the moves used by both players and update the usage counts and effect sums accordingly.
            for side in ("p1", "p2"): # View each side of the battle and check the move details, pokemon state, and opponent state. If a move was used, we update the usage count and effect sums based on the changes in boosts, healing, damage dealt, status inflictions, and other relevant metrics.
                m = t[f"{side}_move_details"]
                if m is None:
                    continue
                name = m["name"]
                uses[name] += 1
                if m["priority"] > 0:
                    sums["priority"][name] += 1
                opp = "p2" if side == "p1" else "p1"
                s = t[f"{side}_pokemon_state"] # Get the current state of the pokemon of current side
                o = t[f"{opp}_pokemon_state"] # Get the current state of the pokemon of the opponent side
                if i > 0:
                    ps = tl[i - 1][f"{side}_pokemon_state"]
                    po = tl[i - 1][f"{opp}_pokemon_state"]
                    if ps["name"] == s["name"]:
                        gain = sum(max(0, s["boosts"][k] - ps["boosts"][k]) for k in BOOST_KEYS)
                        if gain:
                            sums["boost"][name] += gain
                        if s["hp_pct"] > ps["hp_pct"] and s["status"] != "fnt":
                            sums["heal"][name] += s["hp_pct"] - ps["hp_pct"]
                    if po["name"] == o["name"]:
                        drop = max(0.0, po["hp_pct"] - o["hp_pct"])
                        if drop:
                            sums["damage"][name] += drop
                        if (po["status"] != o["status"] and o["status"] not in NO_STATUS):
                            status_sums[o["status"]][name] += 1
                if s["hp_pct"] == 0 or s["status"] == "fnt":
                    sums["self_ko"][name] += 1
                if i + 1 < len(tl):
                    nxt = tl[i + 1][f"{opp}_pokemon_state"]
                    if nxt["name"] != o["name"]:
                        sums["opp_switch"][name] += 1
                        sums["entry_dmg"][name] += max(0.0, 1.0 - nxt["hp_pct"])

    profile: dict[str, dict[str, float]] = {}
    for name, n in uses.items():
        prof = {"uses": float(n)}
        for k in PROFILE_KEYS:
            prof[k] = sums[k][name] / n
        for s in status_sums:
            prof[f"status_{s}"] = status_sums[s][name] / n
        profile[name] = prof
    return profile

priors, gpriors = pokemons_move_priors(train + test)
med_sp = median_pokemons(pokemons)
profile = build_profile(train + test)
#####################

#####################
#6. Features Building

def _static_and_state(d, chart, pokemons):
    """
    Extracts static and state features from a battle dictionary, type chart, and pokemon dictionary, returning a dictionary containing team statistics, lead statistics, battle timeline information, damage taken, healing, fainted rows, switches, status inflictions, move counts, and boost gains for both players.
    We divide the features into six classes:
    static and state features,
    duel features,
    strength features,
    matchup features,
    playstyle features,
    move effectiveness features.
    Collecting around 100 features for each player, we have a total of 200 features.
    The static and state features include team statistics, lead statistics, battle timeline information, damage taken, healing, fainted rows, switches, status inflictions, move counts, and boost gains for both players.
    Used to build the group 1.

    """
    f = {}
    team = d["p1_team_details"]
    lead1, lead2 = team[0], d["p2_lead_details"]

    bsts = [sum(p[f"base_{s}"] for s in STATS) for p in team]
    f["p1_team_bst_mean"] = float(np.mean(bsts))
    f["p1_team_bst_max"] = float(max(bsts))
    f["p1_team_bulk_sum"] = float(sum(p["base_hp"] * (p["base_def"] + p["base_spd"]) / 2 for p in team))
    f["p1_team_off_sum"] = float(sum(max(p["base_atk"], p["base_spa"]) for p in team))
    f["p1_team_spe_100"] = float(sum(1 for p in team if p["base_spe"] >= 100))
    f["p1_team_n_types"] = float(len({t for p in team for t in p["types"] if t != "notype"}))
    f["p1_item_leftovers"] = float(sum(1 for p in team if p["item"] == "leftovers"))
    f["p1_lead_bst"] = float(sum(lead1[f"base_{s}"] for s in STATS))
    f["p2_lead_bst"] = float(sum(lead2[f"base_{s}"] for s in STATS))
    f["lead_bst_diff"] = f["p1_lead_bst"] - f["p2_lead_bst"]
    f["lead_eff_diff"] = (best_stab_eff(chart, lead1["types"], lead2["types"])
                          - best_stab_eff(chart, lead2["types"], lead1["types"]))
    f["lead_spe_diff"] = lead1["base_spe"] - lead2["base_spe"]

    tl = d["battle_timeline"]
    hp1 = {p["name"]: 1.0 for p in team}
    hp2 = {lead2["name"]: 1.0}
    st1 = {p["name"]: "nostatus" for p in team}
    st2 = {lead2["name"]: "nostatus"}
    last_seen2 = {lead2["name"]: 0}
    known1 = {p["name"] for p in team}
    known2 = {lead2["name"]}

    prev_name1 = prev_name2 = None
    prev_hp1 = prev_hp2 = 1.0
    prev_boosts1 = prev_boosts2 = None
    prev_types1 = prev_types2 = None

    dmg_taken1 = dmg_taken2 = 0.0
    heal1 = heal2 = 0.0
    entry1 = entry2 = 0.0
    fnt_rows1 = fnt_rows2 = 0
    switches1 = switches2 = 0
    vol1 = vol2 = 0
    null1 = null2 = 0
    faint_repl1 = faint_repl2 = 0
    last_dmg1 = last_dmg2 = 0
    last_ko1 = last_ko2 = 0
    status_turns1 = {s: 0 for s in STATUSES}
    status_turns2 = {s: 0 for s in STATUSES}
    status_inflict1 = {s: 0 for s in STATUSES}
    status_inflict2 = {s: 0 for s in STATUSES}
    dmg_turn1 = np.zeros(len(tl))
    dmg_turn2 = np.zeros(len(tl))
    tot1_by_turn = np.zeros(len(tl))
    tot2_by_turn = np.zeros(len(tl))
    diff_by_turn = np.zeros(len(tl))

    move_count1 = move_count2 = 0
    move_cat1 = {"physical": 0, "special": 0, "status": 0}
    move_cat2 = {"physical": 0, "special": 0, "status": 0}
    power1 = power2 = 0.0
    effpower1 = effpower2 = 0.0
    expdam1 = expdam2 = 0.0
    se1 = se2 = 0
    res1 = res2 = 0
    best_power1 = best_power2 = 0.0
    known_moves1: set[str] = set()
    known_moves2: set[str] = set()
    move_eff_sum1 = move_eff_n1 = 0.0
    move_eff_sum2 = move_eff_n2 = 0.0
    switch_quality1 = switch_quality2 = 0.0
    boost_gain1 = boost_gain2 = 0.0
    boosts_final1 = {k: 0 for k in BOOST_KEYS}
    boosts_final2 = {k: 0 for k in BOOST_KEYS}

    n = len(tl)
    for idx, t in enumerate(tl):
        s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
        m1, m2 = t["p1_move_details"], t["p2_move_details"]
        n1, n2 = s1["name"], s2["name"]
        if n1 not in hp1:
            hp1[n1] = 1.0
            known1.add(n1)
        if n2 not in hp2:
            hp2[n2] = 1.0
            known2.add(n2)

        sw1 = prev_name1 is not None and n1 != prev_name1
        sw2 = prev_name2 is not None and n2 != prev_name2
        if sw1:
            switches1 += 1
            if m1 is None:
                vol1 += 1
            if st1.get(prev_name1) == "fnt" or prev_hp1 <= 0:
                faint_repl1 += 1
            e = max(0.0, hp1[n1] - s1["hp_pct"])
            entry1 += e
            dmg_taken1 += e
            dmg_turn1[idx] += e
        elif prev_name1 is not None:
            drop = max(0.0, prev_hp1 - s1["hp_pct"])
            gain = max(0.0, s1["hp_pct"] - prev_hp1)
            dmg_taken1 += drop
            heal1 += gain
            dmg_turn1[idx] += drop
            if drop > 0.01:
                last_dmg1 = t["turn"]
        hp1[n1] = max(0.0, s1["hp_pct"])
        st1[n1] = s1["status"]
        if s1["status"] == "fnt":
            fnt_rows1 += 1
            hp1[n1] = 0.0
            last_ko1 = t["turn"]
        for s in STATUSES:
            if s1["status"] == s:
                status_turns1[s] += 1

        if sw2:
            switches2 += 1
            if m2 is None:
                vol2 += 1
            if st2.get(prev_name2) == "fnt" or prev_hp2 <= 0:
                faint_repl2 += 1
            e = max(0.0, hp2[n2] - s2["hp_pct"])
            entry2 += e
            dmg_taken2 += e
            dmg_turn2[idx] += e
        elif prev_name2 is not None:
            drop = max(0.0, prev_hp2 - s2["hp_pct"])
            gain = max(0.0, s2["hp_pct"] - prev_hp2)
            dmg_taken2 += drop
            heal2 += gain
            dmg_turn2[idx] += drop
            if drop > 0.01:
                last_dmg2 = t["turn"]
        hp2[n2] = max(0.0, s2["hp_pct"])
        st2[n2] = s2["status"]
        last_seen2[n2] = t["turn"]
        if s2["status"] == "fnt":
            fnt_rows2 += 1
            hp2[n2] = 0.0
            last_ko2 = t["turn"]
        for s in STATUSES:
            if s2["status"] == s:
                status_turns2[s] += 1

        if m1 is None and not sw1:
            null1 += 1
        if m2 is None and not sw2:
            null2 += 1

        b1, b2 = s1["boosts"], s2["boosts"]
        if prev_boosts1 is not None:
            boost_gain1 += sum(max(0, b1[k] - prev_boosts1[k]) for k in b1)
        if prev_boosts2 is not None:
            boost_gain2 += sum(max(0, b2[k] - prev_boosts2[k]) for k in b2)
        prev_boosts1, prev_boosts2 = dict(b1), dict(b2)
        boosts_final1, boosts_final2 = dict(b1), dict(b2)

        sp1 = pokemons.get(n1, pokemons[lead2["name"]])
        sp2 = pokemons.get(n2, pokemons[lead2["name"]])
        lvl1 = float(next((p["level"] for p in team if p["name"] == n1), lead1["level"]))
        lvl2 = float(next((p["level"] for p in team if p["name"] == n2), lead2["level"]))
        types1 = pokemons.get(n1, {"types": ["notype", "notype"]})["types"]
        types2 = pokemons.get(n2, {"types": ["notype", "notype"]})["types"]
        if n1 == lead1["name"]:
            types1, sp1, lvl1 = lead1["types"], pokemons[lead1["name"]], float(lead1["level"])
        if n2 == lead2["name"]:
            types2, sp2, lvl2 = lead2["types"], pokemons[lead2["name"]], float(lead2["level"])

        if m1 is not None and m1["category"] != "status" and m1["base_power"] > 0:
            move_eff_sum1 += defensive_mult(chart, types2, m1["type"])
            move_eff_n1 += 1
        if m2 is not None and m2["category"] != "status" and m2["base_power"] > 0:
            move_eff_sum2 += defensive_mult(chart, types1, m2["type"])
            move_eff_n2 += 1
        if sw1 and prev_types1 is not None:
            switch_quality1 += best_stab_eff(chart, types1, types2) - best_stab_eff(chart, prev_types1, types2)
        if sw2 and prev_types2 is not None:
            switch_quality2 += best_stab_eff(chart, types2, types1) - best_stab_eff(chart, prev_types2, types1)

        atk1 = stat_value(sp1, "atk" if (m1 and m1["category"] == "physical") else "spa", lvl1)
        atk2 = stat_value(sp2, "atk" if (m2 and m2["category"] == "physical") else "spa", lvl2)
        dfn1 = stat_value(sp1, "def" if (m2 and m2["category"] == "physical") else "spd", lvl1)
        dfn2 = stat_value(sp2, "def" if (m1 and m1["category"] == "physical") else "spd", lvl2)
        hp1max = max_hp_of(sp1, lvl1)
        hp2max = max_hp_of(sp2, lvl2)
        if m1 is not None:
            move_count1 += 1
            known_moves1.add(m1["name"])
            move_cat1[m1["category"]] = move_cat1.get(m1["category"], 0) + 1
            eff, stab, ep = move_expected_power(chart, m1, types1, types2)
            power1 += m1["base_power"]
            effpower1 += ep
            expdam1 += estimate_damage_frac(lvl1, m1["base_power"], atk1, dfn2, hp2max, eff, stab, m1["accuracy"])
            if m1["category"] != "status" and m1["base_power"] > 0:
                if eff > 1:
                    se1 += 1
                elif eff < 1:
                    res1 += 1
                if m1["base_power"] * eff * stab > best_power1:
                    best_power1 = m1["base_power"] * eff * stab
        if m2 is not None:
            move_count2 += 1
            known_moves2.add(m2["name"])
            move_cat2[m2["category"]] = move_cat2.get(m2["category"], 0) + 1
            eff, stab, ep = move_expected_power(chart, m2, types2, types1)
            power2 += m2["base_power"]
            effpower2 += ep
            expdam2 += estimate_damage_frac(lvl2, m2["base_power"], atk2, dfn1, hp1max, eff, stab, m2["accuracy"])
            if m2["category"] != "status" and m2["base_power"] > 0:
                if eff > 1:
                    se2 += 1
                elif eff < 1:
                    res2 += 1
                if m2["base_power"] * eff * stab > best_power2:
                    best_power2 = m2["base_power"] * eff * stab

        tot1 = sum(hp1.values())
        tot2 = sum(hp2.values()) + (6 - len(hp2))
        tot1_by_turn[idx] = tot1
        tot2_by_turn[idx] = tot2
        diff_by_turn[idx] = tot1 - tot2
        prev_name1, prev_hp1 = n1, s1["hp_pct"]
        prev_name2, prev_hp2 = n2, s2["hp_pct"]
        prev_types1, prev_types2 = types1, types2

    seen1: dict[str, str] = {}
    seen2: dict[str, str] = {}
    for t in tl:
        for side, seen, out in (("p1", seen1, status_inflict1), ("p2", seen2, status_inflict2)):
            stt = t[f"{side}_pokemon_state"]
            nm, s = stt["name"], stt["status"]
            old = seen.get(nm, "nostatus")
            if s not in ("nostatus", old) and s != "fnt":
                out[s] += 1
            seen[nm] = s if s != "fnt" else "fnt"

    alive1 = sum(1 for nm in known1 if hp1.get(nm, 0) > 0)
    alive2 = sum(1 for nm in known2 if hp2.get(nm, 0) > 0) + (6 - len(known2))
    tot1, tot2 = sum(hp1.values()), sum(hp2.values()) + (6 - len(hp2))
    f.update({
        "p1_total_hp": tot1, "p2_total_hp": tot2, "total_hp_diff": tot1 - tot2,
        "total_hp_ratio": tot1 / max(1e-6, tot2), "total_hp_sum": tot1 + tot2,
        "p1_alive": float(alive1), "p2_alive": float(alive2), "alive_diff": float(alive1 - alive2),
        "p2_known": float(len(known2)), "p2_unseen": float(6 - len(known2)),
        "p1_fnt_rows": float(fnt_rows1), "p2_fnt_rows": float(fnt_rows2),
        "fnt_rows_diff": float(fnt_rows1 - fnt_rows2),
        "p1_dmg_taken": dmg_taken1, "p2_dmg_taken": dmg_taken2,
        "dmg_taken_diff": dmg_taken1 - dmg_taken2,
        "p1_heal": heal1, "p2_heal": heal2, "heal_diff": heal1 - heal2,
        "p1_entry_dmg": entry1, "p2_entry_dmg": entry2, "entry_dmg_diff": entry1 - entry2,
        "p1_net_dmg": dmg_taken1 - heal1, "p2_net_dmg": dmg_taken2 - heal2,
        "net_dmg_diff": (dmg_taken1 - heal1) - (dmg_taken2 - heal2),
        "p1_switches": float(switches1), "p2_switches": float(switches2),
        "switch_diff": float(switches1 - switches2),
        "p1_vol_switches": float(vol1), "p2_vol_switches": float(vol2),
        "p1_faint_repl": float(faint_repl1), "p2_faint_repl": float(faint_repl2),
        "p1_last_dmg_turn": float(last_dmg1), "p2_last_dmg_turn": float(last_dmg2),
        "p1_last_ko_turn": float(last_ko1), "p2_last_ko_turn": float(last_ko2),
        "p1_switches_last10": float(sum(1 for t in tl[-10:] if t["p1_move_details"] is None)),
        "p2_switches_last10": float(sum(1 for t in tl[-10:] if t["p2_move_details"] is None)),
        "p1_dmg_last10": float(np.sum(dmg_turn1[-10:])), "p2_dmg_last10": float(np.sum(dmg_turn2[-10:])),
        "p1_dmg_first10": float(np.sum(dmg_turn1[:10])), "p2_dmg_first10": float(np.sum(dmg_turn2[:10])),
        "p1_move_count": float(move_count1), "p2_move_count": float(move_count2),
        "p1_move_phys": float(move_cat1["physical"]), "p2_move_phys": float(move_cat2["physical"]),
        "p1_move_spec": float(move_cat1["special"]), "p2_move_spec": float(move_cat2["special"]),
        "p1_move_status": float(move_cat1["status"]), "p2_move_status": float(move_cat2["status"]),
        "p1_power_sum": power1, "p2_power_sum": power2, "power_sum_diff": power1 - power2,
        "p1_effpower_sum": effpower1, "p2_effpower_sum": effpower2,
        "effpower_sum_diff": effpower1 - effpower2,
        "p1_expdam_sum": expdam1, "p2_expdam_sum": expdam2, "expdam_diff": expdam1 - expdam2,
        "p1_se_count": float(se1), "p2_se_count": float(se2),
        "p1_res_count": float(res1), "p2_res_count": float(res2),
        "p1_best_move_power": best_power1, "p2_best_move_power": best_power2,
        "p1_known_moves": float(len(known_moves1)), "p2_known_moves": float(len(known_moves2)),
        "p1_avg_move_eff": move_eff_sum1 / max(1.0, move_eff_n1),
        "p2_avg_move_eff": move_eff_sum2 / max(1.0, move_eff_n2),
        "p1_switch_quality": switch_quality1, "p2_switch_quality": switch_quality2,
        "boost_gain_diff": boost_gain1 - boost_gain2,
        "p1_statused_pokemons": float(len([nm for nm in known1 if st1.get(nm, "nostatus") != "nostatus"])),
        "p2_statused_pokemons": float(len([nm for nm in known2 if st2.get(nm, "nostatus") != "nostatus"])),
        "p1_final_status": float(STATUSES.index(tl[-1]["p1_pokemon_state"]["status"])),
        "p2_final_status": float(STATUSES.index(tl[-1]["p2_pokemon_state"]["status"])),
        "p1_attack_rate": move_count1 / max(1.0, n), "p2_attack_rate": move_count2 / max(1.0, n),
        "p1_switch_rate": switches1 / max(1.0, n), "p2_switch_rate": switches2 / max(1.0, n),
        "p1_dmg_efficiency": dmg_taken2 / max(1e-6, expdam1),
        "p2_dmg_efficiency": dmg_taken1 / max(1e-6, expdam2),
        "p1_wasted_last10": float(sum(1 for j in range(max(1, n - 10), n) if tl[j]["p1_move_details"] is None and tl[j]["p1_pokemon_state"]["name"] == tl[j - 1]["p1_pokemon_state"]["name"])),
        "p2_wasted_last10": float(sum(1 for j in range(max(1, n - 10), n) if tl[j]["p2_move_details"] is None and tl[j]["p2_pokemon_state"]["name"] == tl[j - 1]["p2_pokemon_state"]["name"])),
    })
    for s in ("slp", "par", "tox", "psn", "brn", "frz"):
        f[f"status_turns_diff_{s}"] = float(status_turns1[s] - status_turns2[s])
    for s in ("slp", "par", "tox"):
        f[f"status_inflict_diff_{s}"] = float(status_inflict1.get(s, 0) - status_inflict2.get(s, 0))
    statused1 = float(sum(status_turns1[s] for s in STATUSES if s not in NO_STATUS))
    statused2 = float(sum(status_turns2[s] for s in STATUSES if s not in NO_STATUS))
    f["statused_turns_diff"] = statused1 - statused2
    for k in BOOST_KEYS:
        f[f"boost_diff_{k}"] = float(boosts_final1[k] - boosts_final2[k])
    boost_sum1 = float(sum(boosts_final1.values()))
    boost_sum2 = float(sum(boosts_final2.values()))
    f["boost_sum_diff"] = boost_sum1 - boost_sum2
    f["avg_move_eff_diff"] = f["p1_avg_move_eff"] - f["p2_avg_move_eff"]
    f["switch_quality_diff"] = switch_quality1 - switch_quality2
    f["dmg_diff_last10"] = f["p2_dmg_last10"] - f["p1_dmg_last10"]
    f["dmg_diff_first10"] = f["p2_dmg_first10"] - f["p1_dmg_first10"]
    f["dmg_efficiency_diff"] = f["p1_dmg_efficiency"] - f["p2_dmg_efficiency"]

    x = np.arange(1, n + 1, dtype=float)
    for turn in (5, 10, 15, 20, 25, 30):
        i = min(turn - 1, n - 1)
        f[f"hp_diff_t{turn}"] = float(diff_by_turn[i])
    f["hp_diff_slope"] = float(np.polyfit(x, diff_by_turn, 1)[0])
    f["hp_diff_change_5"] = float(diff_by_turn[-1] - diff_by_turn[-6])
    f["hp_diff_change_10"] = float(diff_by_turn[-1] - diff_by_turn[-11])
    f["hp_diff_min"] = float(np.min(diff_by_turn))
    f["hp_diff_max"] = float(np.max(diff_by_turn))

    last = tl[-1]
    n1, n2 = last["p1_pokemon_state"]["name"], last["p2_pokemon_state"]["name"]
    t1 = lead1["types"] if n1 == lead1["name"] else pokemons.get(n1, {"types": ["notype"]})["types"]
    t2 = lead2["types"] if n2 == lead2["name"] else pokemons.get(n2, {"types": ["notype"]})["types"]
    b1, b2 = last["p1_pokemon_state"]["boosts"], last["p2_pokemon_state"]["boosts"]
    f["active_hp_p1"] = float(last["p1_pokemon_state"]["hp_pct"])
    f["active_hp_p2"] = float(last["p2_pokemon_state"]["hp_pct"])
    f["active_hp_diff"] = f["active_hp_p1"] - f["active_hp_p2"]
    f["active_eff_p1"] = best_stab_eff(chart, t1, t2)
    f["active_eff_p2"] = best_stab_eff(chart, t2, t1)
    f["active_eff_diff"] = f["active_eff_p1"] - f["active_eff_p2"]
    f["active_spe_p1"] = (pokemons.get(n1, pokemons[lead2["name"]])["spe"] * BOOST_MULT[int(b1["spe"])] * (0.25 if last["p1_pokemon_state"]["status"] == "par" else 1.0))
    f["active_spe_p2"] = (pokemons.get(n2, pokemons[lead2["name"]])["spe"] * BOOST_MULT[int(b2["spe"])]  * (0.25 if last["p2_pokemon_state"]["status"] == "par" else 1.0))
    f["active_spe_diff"] = f["active_spe_p1"] - f["active_spe_p2"]
    f["ix_boost_active"] = boost_sum1 * f["active_hp_p1"] - boost_sum2 * f["active_hp_p2"]
    f["p1_active_vs_p2team"] = float(np.mean([best_stab_eff(chart, t1, pokemons.get(nm, {"types": ["notype"]})["types"]) for nm in known2]))
    f["p2_active_vs_p1team"] = float(np.mean([best_stab_eff(chart, t2, p["types"]) for p in team]))
    f["p2_known_bst_sum"] = float(sum(sum(pokemons.get(nm, pokemons[lead2["name"]])[s] for s in STATS) for nm in known2))
    f["p2_known_off_sum"] = float(sum(max(pokemons.get(nm, pokemons[lead2["name"]])["atk"], pokemons.get(nm, pokemons[lead2["name"]])["spa"]) for nm in known2))
    f["p2_known_bulk_sum"] = float(sum(pokemons.get(nm, pokemons[lead2["name"]])["hp"] * (pokemons.get(nm, pokemons[lead2["name"]])["def"] + pokemons.get(nm, pokemons[lead2["name"]])["spd"]) / 2 for nm in known2))
    for pref, sp, stt, lvl in (("p1a", pokemons.get(n1, pokemons[lead1["name"]]), last["p1_pokemon_state"], 100.0), ("p2a", pokemons.get(n2, pokemons[lead2["name"]]), last["p2_pokemon_state"], 100.0)):
        f[f"{pref}_bst"] = float(sum(sp[s] for s in STATS))
        f[f"{pref}_off"] = max(sp["atk"], sp["spa"])
        f[f"{pref}_bulk"] = sp["hp"] * (sp["def"] + sp["spd"]) / 2

    p1_hps = sorted([hp1[p["name"]] for p in team], reverse=True)
    p2_hps = sorted([hp2[nm] for nm in known2] + [1.0] * (6 - len(known2)), reverse=True)
    top2, top3, hp_std, n_full = {}, {}, {}, {}
    for pref, vals in (("p1", p1_hps), ("p2", p2_hps)):
        f[f"{pref}_n_hp_gt_0.5"] = float(sum(v > 0.5 for v in vals))
        f[f"{pref}_n_hp_lt_25"] = float(sum(v < 0.25 for v in vals))
        top2[pref] = float(sum(vals[:2]))
        top3[pref] = float(sum(vals[:3]))
        hp_std[pref] = float(np.std(vals))
        f[f"{pref}_hp_min"] = float(min(vals))
        n_full[pref] = float(sum(v > 0.99 for v in vals))
    f["n_hp_gt_50_diff"] = f["p1_n_hp_gt_0.5"] - f["p2_n_hp_gt_0.5"]
    f["hp_top3_diff"] = top3["p1"] - top3["p2"]
    f["hp_top2_diff"] = top2["p1"] - top2["p2"]
    f["hp_std_diff"] = hp_std["p1"] - hp_std["p2"]
    f["n_full_diff"] = n_full["p1"] - n_full["p2"]

    low = [nm for nm in known2 if nm != n2 and hp2[nm] < 0.2 and last_seen2[nm] < n - 4]
    f["p2_low_hp_gone_0.2"] = float(len(low))
    f["p2_low_hp_gone_hpsum_0.2"] = float(sum(hp2[nm] for nm in low))
    gone, gone_hp = f["p2_low_hp_gone_0.2"], f["p2_low_hp_gone_hpsum_0.2"]
    for w, tag in ((0.5, "05"), (1.0, "10")):
        f[f"hp_diff_corr_{tag}"] = tot1 - (tot2 - w * gone_hp)
    lvl1 = {p["name"]: float(p["level"]) for p in team}
    lvl2 = {lead2["name"]: float(lead2["level"])}
    cands = ko_candidates(tl, "p2", pokemons, lvl2, chart)
    for w, tag in ((0.5, "05"), (0.8, "08")):
        f[f"hp_diff_corr_ko_{tag}"] = tot1 - (tot2 - w * sum(h for _, h, _ in cands))
        if tag == "08":
            f["alive_diff_corr_ko_08"] = float(alive1) - (f["p2_alive"] - w * len(cands))

    if last["p2_move_details"] is not None and last["p2_move_details"]["category"] != "status":
        dmg_to_p1 = move_damage_frac(chart, pokemons, last["p2_move_details"], n2, lvl2.get(n2, 100.0), b2, last["p2_pokemon_state"]["status"], n1, lvl1.get(n1, 100.0), b1)
    else:
        sp2x = pokemons.get(n2, pokemons[lead2["name"]]); sp1x = pokemons.get(n1, pokemons[lead1["name"]])
        a2 = max(stat_value(sp2x, "atk", lvl2.get(n2, 100.0)), stat_value(sp2x, "spa", lvl2.get(n2, 100.0)))
        d1 = (stat_value(sp1x, "def", lvl1.get(n1, 100.0)) if stat_value(sp2x, "atk", lvl2.get(n2, 100.0)) >= stat_value(sp2x, "spa", lvl2.get(n2, 100.0)) else stat_value(sp1x, "spd", lvl1.get(n1, 100.0)))
        dmg_to_p1 = estimate_damage_frac(lvl2.get(n2, 100.0), 80.0, a2, d1, max_hp_of(sp1x, lvl1.get(n1, 100.0)), best_stab_eff(chart, t2, t1), 1.5, 1.0)
    if last["p1_move_details"] is not None and last["p1_move_details"]["category"] != "status":
        dmg_to_p2 = move_damage_frac(chart, pokemons, last["p1_move_details"], n1, lvl1.get(n1, 100.0), b1, last["p1_pokemon_state"]["status"], n2, lvl2.get(n2, 100.0), b2)
    else:
        sp1x = pokemons.get(n1, pokemons[lead1["name"]]); sp2x = pokemons.get(n2, pokemons[lead2["name"]])
        a1 = max(stat_value(sp1x, "atk", lvl1.get(n1, 100.0)), stat_value(sp1x, "spa", lvl1.get(n1, 100.0)))
        d2 = (stat_value(sp2x, "def", lvl2.get(n2, 100.0)) if stat_value(sp1x, "atk", lvl1.get(n1, 100.0)) >= stat_value(sp1x, "spa", lvl1.get(n1, 100.0)) else stat_value(sp2x, "spd", lvl2.get(n2, 100.0)))
        dmg_to_p2 = estimate_damage_frac(lvl1.get(n1, 100.0), 80.0, a1, d2, max_hp_of(sp2x, lvl2.get(n2, 100.0)), best_stab_eff(chart, t1, t2), 1.5, 1.0)
    f["p1_active_ttk"] = min(20.0, f["active_hp_p1"] / max(1e-3, dmg_to_p1))
    f["p2_active_ttk"] = min(20.0, f["active_hp_p2"] / max(1e-3, dmg_to_p2))
    f["active_ttk_diff"] = f["p1_active_ttk"] - f["p2_active_ttk"]
    return f
def _duel(st, chart, med_sp):
    """
    Compute duel features for the active Pokemon of player 1 against the roster of player 2.
    Estimates who has the better matchups at turn 30 by simulating 1-vs-1 duels.
    The active pokemon of each player is matched against the opponent's active one. For each duel it computes the damages in both ways and declare the winner based on the number of turns to KO the opponent (fewer turns). The duel is repeated for each Pokemon in the opponent's roster and the results are aggregated.
    Returns a dictionary of features related to the duel outcomes, including win/loss counts, damage margins, and speed advantages for P1 and P2.
    Used to build the group 2.
    """
    f = {}
    tl = st["tl"]
    last = tl[-1]
    n1 = last["p1_pokemon_state"]["name"]
    n2 = last["p2_pokemon_state"]["name"]
    b1 = last["p1_pokemon_state"]["boosts"]
    b2 = last["p2_pokemon_state"]["boosts"]
    sst1 = last["p1_pokemon_state"]["status"]
    sst2 = last["p2_pokemon_state"]["status"]
    sp1 = st["p1_info"][n1]["sp"]; lv1 = st["p1_info"][n1]["level"]
    sp2 = st["p2_info"][n2]["sp"]; lv2 = st["p2_info"][n2]["level"]

    def duel(att_sp, att_lvl, att_boosts, att_status, def_sp, def_lvl, def_boosts, def_status, def_hp_frac=1.0):
        """
        Active function that works according to the _duel description, to clear the code.
        """
        eff = best_stab_eff(chart, att_sp["types"], def_sp["types"])
        atk_phys = stat_value(att_sp, "atk", att_lvl) * BOOST_MULT[int(att_boosts["atk"])]
        atk_spec = stat_value(att_sp, "spa", att_lvl) * BOOST_MULT[int(att_boosts["spa"])]
        use_phys = atk_phys >= atk_spec
        atk = atk_phys * (0.5 if att_status == "brn" else 1.0) if use_phys else atk_spec
        dfn = stat_value(def_sp, "def" if use_phys else "spd", def_lvl) * \
            BOOST_MULT[int(def_boosts["def" if use_phys else "spd"])]
        dmg = min(1.5, max(1e-4, estimate_damage_frac(att_lvl, 80.0, atk, dfn,
                                                       max_hp_of(def_sp, def_lvl), eff, 1.5, 1.0)))
        turns_att = int(np.ceil(max(0.05, def_hp_frac) / dmg))
        eff2 = best_stab_eff(chart, def_sp["types"], att_sp["types"])
        atk2_phys = stat_value(def_sp, "atk", def_lvl) * BOOST_MULT[int(def_boosts["atk"])]
        atk2_spec = stat_value(def_sp, "spa", def_lvl) * BOOST_MULT[int(def_boosts["spa"])]
        use_phys2 = atk2_phys >= atk2_spec
        atk2 = atk2_phys * (0.5 if def_status == "brn" else 1.0) if use_phys2 else atk2_spec
        dfn2 = stat_value(att_sp, "def" if use_phys2 else "spd", att_lvl) * \
            BOOST_MULT[int(att_boosts["def" if use_phys2 else "spd"])]
        dmg2 = min(1.5, max(1e-4, estimate_damage_frac(def_lvl, 80.0, atk2, dfn2,
                                                        max_hp_of(att_sp, att_lvl), eff2, 1.5, 1.0)))
        att_hp = max(0.05, float(att_sp.get("_hp", 1.0)))
        turns_def = int(np.ceil(att_hp / dmg2))
        spe_att = stat_value(att_sp, "spe", att_lvl) * BOOST_MULT[int(att_boosts["spe"])] * (0.25 if att_status == "par" else 1.0)
        spe_def = stat_value(def_sp, "spe", def_lvl) * BOOST_MULT[int(def_boosts["spe"])] * (0.25 if def_status == "par" else 1.0)
        faster = spe_att >= spe_def
        return dict(win=(turns_att < turns_def) or (turns_att == turns_def and faster), dmg=dmg, dmg2=dmg2, margin=turns_def - turns_att, speed_edge=spe_att - spe_def)

    no_boosts = {k: 0 for k in BOOST_KEYS}
    p2_roster = []
    for nm, dd in st["p2_info"].items():
        h = st["hp2"].get(nm, 1.0)
        s_p = {**dd["sp"], "_hp": float(h)}
        p2_roster.append((nm, s_p, dd["level"], st["st2"].get(nm, "nostatus")))
    for _ in range(6 - len(st["p2_info"])):
        p2_roster.append(("<unseen>", {**med_sp, "_hp": 1.0}, 100.0, "nostatus"))
    wins = [duel(sp1, lv1, b1, sst1, s_p, lvl, no_boosts, status) for _, s_p, lvl, status in p2_roster]
    f["p1a_wins_vs_p2roster"] = float(sum(w["win"] for w in wins))
    f["p1a_loses_vs_p2roster"] = float(sum(not w["win"] for w in wins))
    f["p1a_win_margin_sum"] = float(sum(w["margin"] for w in wins))
    f["p1a_ohko_count"] = float(sum(1 for (_, s_p, _, _), w in zip(p2_roster, wins) if w["dmg"] >= s_p["_hp"]))
    f["p1a_speed_beat_count"] = float(sum(1 for w in wins if w["speed_edge"] > 0))

    p1_roster = []
    for nm, dd in st["p1_info"].items():
        h = st["hp1"].get(nm, 1.0)
        p1_roster.append((nm, {**dd["sp"], "_hp": float(h)}, dd["level"], st["st1"].get(nm, "nostatus")))
    wins2 = [duel(sp2, lv2, b2, sst2, s_p, lvl, no_boosts, status) for _, s_p, lvl, status in p1_roster]
    f["p2a_wins_vs_p1team"] = float(sum(w["win"] for w in wins2))
    f["p2a_loses_vs_p1team"] = float(sum(not w["win"] for w in wins2))
    f["p2a_win_margin_sum"] = float(sum(w["margin"] for w in wins2))
    f["p2a_ohko_count"] = float(sum(1 for (_, s_p, _, _), w in zip(p1_roster, wins2) if w["dmg"] >= s_p["_hp"]))
    f["p2a_speed_beat_count"] = float(sum(1 for w in wins2 if w["speed_edge"] > 0))
    f["sweep_win_diff"] = f["p1a_wins_vs_p2roster"] - f["p2a_wins_vs_p1team"]
    f["sweep_margin_diff"] = f["p1a_win_margin_sum"] - f["p2a_win_margin_sum"]

    b1s = []
    for nm, dd in st["p1_info"].items():
        if nm == n1 or st["hp1"].get(nm, 0) <= 0:
            continue
        s_p = {**dd["sp"], "_hp": float(st["hp1"][nm])}
        b1s.append(duel(s_p, dd["level"], no_boosts, st["st1"].get(nm, "nostatus"), sp2, lv2, b2, sst2, def_hp_frac=float(st["hp2"].get(n2, 1.0)))["margin"])
    b2s = []
    for nm, dd in st["p2_info"].items():
        if nm == n2 or st["hp2"].get(nm, 0) <= 0:
            continue
        s_p = {**dd["sp"], "_hp": float(st["hp2"][nm])}
        b2s.append(duel(s_p, dd["level"], no_boosts, st["st2"].get(nm, "nostatus"), sp1, lv1, b1, sst1, def_hp_frac=float(st["hp1"].get(n1, 1.0)))["margin"])
    f["p1_best_bench_margin_vs_p2active"] = max(b1s) if b1s else -10.0
    f["p2_best_bench_margin_vs_p1active"] = max(b2s) if b2s else -10.0
    f["bench_answer_diff"] = f["p1_best_bench_margin_vs_p2active"] - f["p2_best_bench_margin_vs_p1active"]

    def resist(roster, types):
        return sum(1 for _, sp, _, _ in roster if any(defensive_mult(chart, sp["types"], t) < 1 for t in types if t != "notype"))
    f["resist_diff"] = float(resist(p1_roster, sp2["types"]) - resist(p2_roster, sp1["types"]))
    return f
def _strength(st):
    """
    Compute strength features based on the remaining Pokemon in each player's roster.
    Returns a dictionary of features related to the remaining strength of each player's team,including total base stat sums, offensive and defensive capabilities, and counts of mid-tier Pokemon.
    Used to build the group 3.
    """
    f = {}
    rem_off, rem_bulk, n_mid = {}, {}, {}
    for side in ("p1", "p2"):
        info = st[f"{side}_info"]
        hp = st[f"hp{'1' if side == 'p1' else '2'}"]
        vals = []
        for nm, dd in info.items():
            h = hp.get(nm, 0.0)
            sp = dd["sp"]
            vals.append((h, max(sp["atk"], sp["spa"]),
                         sp["hp"] * (sp["def"] + sp["spd"]) / 2.0,
                         sum(sp[s] for s in STATS)))
        unseen = 6 - len(info) if side == "p2" else 0
        if unseen:
            if vals:
                ao = float(np.mean([v[1] for v in vals]))
                ab = float(np.mean([v[2] for v in vals]))
                ast = float(np.mean([v[3] for v in vals]))
            else:
                ao, ab, ast = 100.0, 300.0, 500.0
            vals += [(1.0, ao, ab, ast)] * unseen
        f[f"str_{side}_rem_bst"] = float(sum(h * b for h, _, _, b in vals))
        rem_off[side] = float(sum(h * o for h, o, _, _ in vals))
        rem_bulk[side] = float(sum(h * b for h, _, b, _ in vals))
        n_mid[side] = float(sum(1 for h, _, _, _ in vals if h > 0.5))
    f["str_diff_rem_bst"] = f["str_p1_rem_bst"] - f["str_p2_rem_bst"]
    f["str_diff_rem_off"] = rem_off["p1"] - rem_off["p2"]
    f["str_diff_rem_bulk"] = rem_bulk["p1"] - rem_bulk["p2"]
    f["str_diff_n_mid"] = n_mid["p1"] - n_mid["p2"]
    f["str_ratio_bst"] = f["str_p1_rem_bst"] / max(1e-3, f["str_p2_rem_bst"])
    f["str_ratio_off"] = rem_off["p1"] / max(1e-3, rem_off["p2"])
    f["str_ratio_bulk"] = rem_bulk["p1"] / max(1e-3, rem_bulk["p2"])
    return f
def _matchup(st, chart, priors, gpriors, med_sp):
    """
    Compute duel features for the active Pokemon of player 1 against the roster of player 2.
    Estimates who has the better matchups at turn 30 by simulating 1-vs-All duels.
    The active pokemon of each player is matched against all the opponent's pokemon. For each pair it computes the damages in both ways and declare the winner based on the number of turns to KO the opponent (fewer turns). The duel is repeated for each Pokemon in the opponent's roster and the results are aggregated.
    Returns a dictionary of features related to the duel outcomes, including win/loss counts, damage margins, and speed advantages for P1 and P2.
    Used to build the group 4.
    """
    f = {}
    tl = st["tl"]
    last = tl[-1]
    n1 = last["p1_pokemon_state"]["name"]
    n2 = last["p2_pokemon_state"]["name"]
    b1 = last["p1_pokemon_state"]["boosts"]
    b2 = last["p2_pokemon_state"]["boosts"]
    sst1 = last["p1_pokemon_state"]["status"]
    sst2 = last["p2_pokemon_state"]["status"]

    def roster(side):
        """
        Active function that returns the roster of a player with relevant information for matchup calculations. Follow the logic according to the specs of the _matchup function.
        """
        info = st[f"{side}_info"]
        hp = st[f"hp{'1' if side == 'p1' else '2'}"]
        stt = st[f"st{'1' if side == 'p1' else '2'}"]
        act = n1 if side == "p1" else n2
        out = []
        for nm, dd in info.items():
            h = float(hp.get(nm, 0.0))
            if h <= 0:
                continue
            out.append(dict(name=nm, sp=dd["sp"], hp=h, item=dd.get("item"), boosts=(b1 if side == "p1" else b2) if nm == act else {k: 0 for k in BOOST_KEYS}, status=(sst1 if side == "p1" else sst2) if nm == act else stt.get(nm, "nostatus"), active=(nm == act)))

        if side == "p2":
            for _ in range(6 - len(info)):
                out.append(dict(name="<unseen>", sp=med_sp, hp=1.0, item=None, boosts={k: 0 for k in BOOST_KEYS}, status="nostatus", active=False))
        return out

    p1r, p2r = roster("p1"), roster("p2")

    def pair(a, b):
        pa = gpriors if a["name"] == "<unseen>" else priors.get(a["name"], gpriors)
        pb = gpriors if b["name"] == "<unseen>" else priors.get(b["name"], gpriors)
        da, ba, ca, _ = expected_damage(chart, a["sp"], b["sp"], pa, a["boosts"], a["status"],
                                        b["boosts"], b["status"])
        db, bb, cb, _ = expected_damage(chart, b["sp"], a["sp"], pb, b["boosts"], b["status"],
                                        a["boosts"], a["status"])
        ta = min(60.0, b["hp"] / max(da, 1e-4))
        tb = min(60.0, a["hp"] / max(db, 1e-4))
        sa = stat_value(a["sp"], "spe", 100.0) * BOOST_MULT[int(a["boosts"].get("spe", 0))] * (0.25 if a["status"] == "par" else 1.0)
        sb = stat_value(b["sp"], "spe", 100.0) * BOOST_MULT[int(b["boosts"].get("spe", 0))] * (0.25 if b["status"] == "par" else 1.0)
        margin = float(np.clip(tb - ta - (0.0 if sa >= sb else 1.0), -20.0, 20.0))
        return dict(dmg_ab=da, dmg_ba=db, margin=margin, win=margin > 0, ohko_a=ba >= b["hp"], ohko_b=bb >= a["hp"], cover_ab=ca, cover_ba=cb)

    wins1 = wins2 = ohko1 = ohko2 = 0
    cover1 = cover2 = dmg1 = dmg2 = 0.0
    best1, best2 = [], []
    for a in p1r:
        ms = []
        for b in p2r:
            p = pair(a, b)
            ms.append(p["margin"]); wins1 += int(p["win"]); ohko1 += int(p["ohko_a"]); ohko2 += int(p["ohko_b"])
            cover1 += p["cover_ab"]; cover2 += p["cover_ba"]
            dmg1 += min(b["hp"], p["dmg_ab"]); dmg2 += min(a["hp"], p["dmg_ba"])
        if ms:
            best1.append((a["hp"], max(ms), sum(1 for m in ms if m > 0)))
    for b in p2r:
        ms = [pair(a, b)["margin"] for a in p1r]
        wins2 += sum(1 for m in ms if m < 0)
        if ms:
            best2.append((b["hp"], max(-m for m in ms), sum(1 for m in ms if m < 0)))
    f["mu_p1_wins"] = float(wins1)
    f["mu_p2_wins"] = float(wins2)
    f["mu_wins_diff"] = float(wins1 - wins2)
    f["mu_wins_ratio"] = (wins1 + 0.5) / (wins2 + 0.5)
    f["mu_ohko_diff"] = float(ohko1 - ohko2)
    f["mu_cover_diff"] = float(cover1 - cover2)
    f["mu_dmg_out_diff"] = float(dmg1 - dmg2)
    f["mu_sweep_diff"] = float(max((x[2] for x in best1), default=0) - max((x[2] for x in best2), default=0))
    f["mu_best_margin_hp_diff"] = float(sum(h * m for h, m, _ in best1) - sum(h * m for h, m, _ in best2))
    return f
def _behavior(d):
    """
    Compute behavior features based on the battle timeline.
    For each side of the battle, it counts the number of voluntary and forced switches, reactive and proactive switches, phazing moves, moves that cannot be executed, status hits, setup turns, boost gains and losses, low HP turns, switch-ins with full HP or after being hit, and the turn of the first KO. Using to collect the behavior of the players.
    Used to build the group 5.
    """
    f = {}
    tl = d["battle_timeline"]
    for side in ("p1", "p2"):
        opp = "p2" if side == "p1" else "p1"
        prev_name = None; prev_hp = 1.0; prev_status = "nostatus"; prev_boosts = None
        vol = forced = reactive = proactive = phazed = cant = setup = 0
        sw_full = sw_hit = 0
        boost_gain = boost_loss = 0.0
        status_hits = 0
        low_hp = 0
        first_ko = len(tl) + 1
        prev_opp_status = tl[0][f"{opp}_pokemon_state"]["status"]
        prev_opp_name = tl[0][f"{opp}_pokemon_state"]["name"]
        for t in tl:
            s, o = t[f"{side}_pokemon_state"], t[f"{opp}_pokemon_state"]
            m = t[f"{side}_move_details"]
            nm = s["name"]; hp = 0.0 if s["status"] == "fnt" or s["hp_pct"] == 0 else float(s["hp_pct"])
            switched = prev_name is not None and nm != prev_name
            if switched:
                is_forced = prev_status == "fnt" or prev_hp <= 0
                if is_forced:
                    forced += 1
                else:
                    vol += 1
                    if prev_hp <= 0.5: reactive += 1
                    elif prev_hp >= 0.8: proactive += 1
                if hp >= 0.99:
                    sw_full += 1
                else:
                    sw_hit += 1
                if prev_boosts is not None:
                    boost_loss += sum(max(0, v) for v in prev_boosts.values())
                if t[f"{opp}_move_details"] is not None and t[f"{opp}_move_details"]["category"] == "status":
                    phazed += 1
            elif m is None:
                cant += 1
            if (o["status"] != prev_opp_status and o["status"] not in NO_STATUS
                    and o["name"] == prev_opp_name):
                status_hits += 1
            if prev_boosts is not None and not switched:
                gain = sum(max(0, s["boosts"][k] - prev_boosts[k]) for k in BOOST_KEYS)
                if gain: setup += 1; boost_gain += gain
            if 0 < hp < 0.3: low_hp += 1
            if (o["hp_pct"] == 0 or o["status"] == "fnt") and not (
                    prev_opp_name == o["name"] and prev_opp_status == "fnt") and first_ko > len(tl):
                first_ko = t["turn"]
            prev_name, prev_hp, prev_status = nm, hp, s["status"]
            prev_boosts = dict(s["boosts"]); prev_opp_status = o["status"]
            prev_opp_name = o["name"]
        f[f"beh_{side}_vol_switch"] = float(vol)
        f[f"beh_{side}_forced_switch"] = float(forced)
        f[f"beh_{side}_reactive"] = float(reactive)
        f[f"beh_{side}_proactive"] = float(proactive)
        f[f"beh_{side}_phazed"] = float(phazed)
        f[f"beh_{side}_cant_move"] = float(cant)
        f[f"beh_{side}_status_hits"] = float(status_hits)
        f[f"beh_{side}_status_hit_rate"] = status_hits / max(1.0, len(tl))
        f[f"beh_{side}_setup_turns"] = float(setup)
        f[f"beh_{side}_boost_gain"] = boost_gain
        f[f"beh_{side}_boost_reset_loss"] = boost_loss
        f[f"beh_{side}_low_hp_turns"] = float(low_hp)
        f[f"beh_{side}_switch_in_full"] = float(sw_full)
        f[f"beh_{side}_switch_in_after_hit"] = float(sw_hit)
        f[f"beh_{side}_first_ko_turn"] = float(first_ko)
    for k in ("vol_switch", "forced_switch", "reactive", "proactive", "phazed", "cant_move", "status_hits", "status_hit_rate", "setup_turns", "boost_gain", "boost_reset_loss"):
        f[f"beh_diff_{k}"] = f[f"beh_p1_{k}"] - f[f"beh_p2_{k}"]
    f["beh_first_blood_p1"] = float(f["beh_p1_first_ko_turn"] < f["beh_p2_first_ko_turn"])
    f["beh_first_blood_p2"] = float(f["beh_p2_first_ko_turn"] < f["beh_p1_first_ko_turn"])
    return f
def _me(d, profile):
    """
    Compute Move Effectiveness (ME) features based on the moves used by each player and their effectiveness profiles.
    For each side of the battle, it counts the number of times each move was used and calculates the expected effectiveness of those moves based on a provided profile.
    Used to build the group 6.
    """
    counts = {"p1": Counter(), "p2": Counter()}
    for t in d["battle_timeline"]:
        for side in ("p1", "p2"):
            m = t[f"{side}_move_details"]
            if m is not None:
                counts[side][m["name"]] += 1
    keys = ("boost", "heal", "self_ko", "opp_switch", "entry_dmg", "damage", "status_slp", "status_par", "status_tox")
    f = {}
    side_vals = {}
    for side in ("p1", "p2"):
        side_vals[side] = {k: float(sum(c * profile.get(nm, {}).get(k, 0.0)
                                        for nm, c in counts[side].items()))
                           for k in keys}
        for k in ("opp_switch", "entry_dmg"):
            f[f"me_{side}_{k}"] = side_vals[side][k]
    for k in keys:
        f[f"me_diff_{k}"] = side_vals["p1"][k] - side_vals["p2"][k]
    return f
def extract_curated(d, chart, pokemons, priors, gpriors, med_sp, profile):
    """
    Extract curated features from a single battle dictionary.
    This function combines static features, battle state features, duel features, strength features, matchup features, behavior features, and move effectiveness features into a single dictionary of curated features for a given battle.
    """
    f = _static_and_state(d, chart, pokemons)
    st = battle_state(d, pokemons, chart)
    f.update(_duel(st, chart, med_sp))
    f.update(_strength(st))
    f.update(_matchup(st, chart, priors, gpriors, med_sp))
    f.update(_behavior(d))
    f.update(_me(d, profile))
    f["ix_hp_alive"] = f["total_hp_diff"] * f["alive_diff"]
    f["ix_hp_status"] = f["total_hp_diff"] * f["statused_turns_diff"]
    f["ix_hp_boost"] = f["total_hp_diff"] * f["boost_sum_diff"]
    f["ix_hp_unseen"] = f["total_hp_diff"] * f["p2_unseen"]
    f["ix_activehp_eff"] = f["active_hp_diff"] * f["active_eff_diff"]
    f["ix_activehp_spe"] = f["active_hp_diff"] * f["active_spe_diff"]
    f["ix_hp_dmg10"] = f["total_hp_diff"] * f["dmg_diff_last10"]
    f["ix_sweep_hp"] = f["total_hp_diff"] * f["sweep_win_diff"]
    f["ix_hp_lowgone"] = f["total_hp_diff"] * f["p2_low_hp_gone_0.2"]
    f["ix_alive_status"] = f["alive_diff"] * f["statused_turns_diff"]
    return f
def build_curated():
    """
    Build curated features for the training and test datasets.
    Returning all the features in a DataFrame along with the target variable and test IDs, to create the feature matrices.
    """

    def make(data, agg):
        """
        Takes all the features extracted from the battle data and aggregates them into a DataFrame, adding additional features related to hidden Pokemon and their HP.
        """
        X = pd.DataFrame([extract_curated(d, chart, pokemons, priors, gpriors, med_sp, profile) for d in data])
        X["ko_hidden_prob"] = agg["ko_hidden_prob"].to_numpy()
        X["ko_hidden_hp"] = agg["ko_hidden_hp"].to_numpy()
        X["p2_alive_ko"] = X["p2_alive"] - X["ko_hidden_prob"]
        X["alive_diff_ko"] = X["p1_alive"] - X["p2_alive_ko"]
        X["p2_hp_ko"] = X["p2_total_hp"] - X["ko_hidden_hp"]
        X["hp_diff_ko"] = X["p1_total_hp"] - X["p2_hp_ko"]
        return X

    X_train = make(train, agg_tr)
    X_test = make(test, agg_te)
    y_train = np.array([d["player_won"] for d in train])
    ids = np.array([d["battle_id"] for d in test])
    assert X_train.shape[1] == 276, X_train.shape
    assert list(X_train.columns) == list(X_test.columns)
    return X_train, y_train, X_test, ids

#####################

#####################
#7. Optuna Hyperparameter Tuning

t0 = time.time()
X_train, y_train, X_test, test_ids = build_curated()
print("train", X_train.shape, "| test", X_test.shape, f"({time.time() - t0:.0f}s)")

optuna.logging.set_verbosity(optuna.logging.WARNING)

inner_folds_tuning = list(
    StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    ).split(X_train, y_train)
)

def print_trial_callback(study, trial):
    """
    Prints the log-loss of each Optuna trial and flags when a new best is found.
    """
    duration = trial.duration.total_seconds() if trial.duration else 0.0

    if trial.state == TrialState.COMPLETE:
        is_best = study.best_trial.number == trial.number
        marker = " ★ new best" if is_best else ""
        print(f"\tTrial {trial.number:>2d} logloss: {trial.value:.5f} "
              f"(Best: {study.best_value:.5f}){marker} - {duration:.1f}s")
    elif trial.state == TrialState.PRUNED:
        print(f"\tTrial {trial.number:>2d} PRUNED - {duration:.1f}s")
    elif trial.state == TrialState.FAIL:
        print(f"\tTrial {trial.number:>2d} FAILED")
def objective_xgb(trial):
    """
    Objective function for XGBoost hyperparameter optimization.
    Use the 3-fold log-loss of XGBoost.
    """
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 500, 1500, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
    )
    scores = []
    for step, (tr, va) in enumerate(inner_folds_tuning):
        m = XGBClassifier(
            **params,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
        )
        m.fit(X_train.iloc[tr], y_train[tr])
        s = log_loss(y_train[va], m.predict_proba(X_train.iloc[va])[:, 1])
        scores.append(s)

        current_mean = float(np.mean(scores))

        trial.report(current_mean, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))
def objective_cb(trial):
    """
    Objective function for CatBoost hyperparameter optimization.
    Use the 3-fold log-loss of CatBoost.
    """
    params = dict(
        iterations=trial.suggest_int("iterations", 500, 1500, step=100),
        depth=trial.suggest_int("depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 2.0, 15.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        rsm=trial.suggest_float("rsm", 0.5, 1.0),
    )
    scores = []
    for step, (tr, va) in enumerate(inner_folds_tuning):
        m = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="Logloss",
            thread_count=-1,
            random_seed=SEED,
            verbose=0
        )
        m.fit(X_train.iloc[tr], y_train[tr])
        s = log_loss(y_train[va], m.predict_proba(X_train.iloc[va])[:, 1])
        scores.append(s)

        current_mean = float(np.mean(scores))

        trial.report(current_mean, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))
def objective_lr(trial):
    """
    Objective function for Logistic Regression hyperparameter optimization.
    Use the 3-fold log-loss of LR.
    """
    C = trial.suggest_float("C", 1e-4, 1e0, log=True)
    scores = []
    for step, (tr, va) in enumerate(inner_folds_tuning):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=5000),
        )
        m.fit(X_train.iloc[tr], y_train[tr])
        s = log_loss(y_train[va], m.predict_proba(X_train.iloc[va])[:, 1])
        scores.append(s)
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))
def run_study(objective, name, n_trials, timeout=None):
    """
    Runs an Optuna study for one model and returns it.
    """
    print(f"\nOptimization Study for {name.upper()} with {n_trials} trials")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name=name,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        callbacks=[print_trial_callback]
    )

    elapsed = (time.time() - t0) / 60
    pruned_trials = len(study.get_trials(states=[TrialState.PRUNED]))
    complete_trials = len(study.get_trials(states=[TrialState.COMPLETE]))

    print(f"\nResults for {name.upper()}")
    print(f"Completed trials: {complete_trials}")
    print(f"Pruned trials: {pruned_trials}")
    print(f"Best Trial: #{study.best_trial.number} (logloss: {study.best_value:.5f})")
    print(f"Elapsed: {elapsed:.2f} min")
    print(f"Best Params: {study.best_params}\n")

    return study
# Try all models to find the best parameters
study_lr  = run_study(objective_lr,  "lr",  n_trials=30)
study_xgb = run_study(objective_xgb, "xgb", n_trials=40, timeout=60*60)
study_cb  = run_study(objective_cb,  "cb",  n_trials=40, timeout=60*60)

#####################

#####################
#8. Cross-Validation and Final Model Training

"""
Best hyperparameters from Optuna optimization.
"""
best_logreg_c = float(study_lr.best_params["C"])
best_xgb_params = study_xgb.best_params
best_catboost_params = study_cb.best_params

NUM_FOLDS = 10
cv_folds = list(
    StratifiedKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=SEED
    ).split(X_train, y_train)
)

"""
Hyperparameter configurations for XGBoost and CatBoost (Hand-tuned + Optuna).
"""
xgb_params_depth4 = dict(
    n_estimators=900,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_lambda=3.0,
    min_child_weight=10,
    reg_alpha=0.5,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)
xgb_params_depth3 = {
    **xgb_params_depth4,
    "max_depth": 3,
    "n_estimators": 1000,
    "random_state": SEED,
}
xgb_params_optuna = {
    **best_xgb_params,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}

catboost_params_depth4 = dict(
    iterations=1200,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=5.0,
    subsample=0.85,
    rsm=0.7,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=SEED,
    thread_count=-1,
    verbose=0,
)
catboost_params_depth5 = {
    **catboost_params_depth4,
    "depth": 5,
    "iterations": 1000,
    "l2_leaf_reg": 7.0,
    "random_seed": 2,
}
catboost_params_optuna = {
    **best_catboost_params,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "thread_count": -1,
    "verbose": 0,
    "random_seed": SEED,
}


def train_and_eval_logreg(reg_strength_c, model_label=""):
    """
    Run Logistic Regression with cross-validation and final refit on the entire training set.
    10-fold OOF prediction and test predictions of a standard LR with the C input.
    """
    start_time = time.time()
    oof_predictions = np.zeros(len(y_train))
    cv_test_predictions = np.zeros(len(X_test))

    for train_idx, val_idx in cv_folds:
        fold_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=reg_strength_c, max_iter=5000, random_state=SEED
            ),
        )
        fold_model.fit(X_train.iloc[train_idx], y_train[train_idx])
        oof_predictions[val_idx] = fold_model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]
        cv_test_predictions += (
            fold_model.predict_proba(X_test)[:, 1] / NUM_FOLDS
        )

    # Refit on the entire training dataset
    full_train_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=reg_strength_c, max_iter=5000, random_state=SEED
        ),
    )
    full_train_model.fit(X_train, y_train)
    full_test_predictions = full_train_model.predict_proba(X_test)[:, 1]

    # Blend 50% CV predictions and 50% full-retrain predictions
    final_test_predictions = 0.5 * (cv_test_predictions + full_test_predictions)

    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_train, oof_predictions > 0.5)
    loss = log_loss(y_train, oof_predictions)
    print(f"\tLR {model_label:<12}: acc={accuracy:.4f}, logloss={loss:.4f}  ({elapsed_time:.1f}s)", flush=True)

    return oof_predictions, final_test_predictions
def train_and_eval_xgb(xgb_params, model_label=""):
    """
    10-fold OOF predictions and test prediction using XGBoost, with the given parameters.
    """
    start_time = time.time()
    oof_predictions = np.zeros(len(y_train))
    cv_test_predictions = np.zeros(len(X_test))

    for train_idx, val_idx in cv_folds:
        fold_model = XGBClassifier(**xgb_params)
        fold_model.fit(X_train.iloc[train_idx], y_train[train_idx])
        oof_predictions[val_idx] = fold_model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]
        cv_test_predictions += (
            fold_model.predict_proba(X_test)[:, 1] / NUM_FOLDS
        )

    full_train_model = XGBClassifier(**xgb_params)
    full_train_model.fit(X_train, y_train)
    full_test_predictions = full_train_model.predict_proba(X_test)[:, 1]

    final_test_predictions = 0.5 * (cv_test_predictions + full_test_predictions)

    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_train, oof_predictions > 0.5)
    loss = log_loss(y_train, oof_predictions)
    print(f"\tXGB {model_label:<11}: acc={accuracy:.4f}, logloss={loss:.4f} ({elapsed_time:.1f}s)", flush=True)

    return oof_predictions, final_test_predictions
def train_and_eval_catboost(cb_params, model_label=""):
    """
    10-fold OOF predictions and test predictions of CatBoost, using the given parameters.
    """
    start_time = time.time()
    oof_predictions = np.zeros(len(y_train))
    cv_test_predictions = np.zeros(len(X_test))

    for train_idx, val_idx in cv_folds:
        fold_model = CatBoostClassifier(**cb_params)
        fold_model.fit(X_train.iloc[train_idx], y_train[train_idx])
        oof_predictions[val_idx] = fold_model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]
        cv_test_predictions += (
            fold_model.predict_proba(X_test)[:, 1] / NUM_FOLDS
        )

    full_train_model = CatBoostClassifier(**cb_params)
    full_train_model.fit(X_train, y_train)
    full_test_predictions = full_train_model.predict_proba(X_test)[:, 1]

    final_test_predictions = 0.5 * (cv_test_predictions + full_test_predictions)

    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_train, oof_predictions > 0.5)
    loss = log_loss(y_train, oof_predictions)
    print(f"\tCB {model_label:<12}: acc={accuracy:.4f}, logloss={loss:.4f} ({elapsed_time:.1f}s)", flush=True)

    return oof_predictions, final_test_predictions
"""
1. Train Logistic Regression models with different C values and average their predictions.
"""
logreg_oof_list, logreg_test_list = [], []
c_values_to_test = [0.003, 0.01, 0.05, best_logreg_c]

for idx, c_value in enumerate(c_values_to_test):
    label = f"Opt C={c_value:.4f}" if idx == 3 else f"C={c_value}"
    model_oof, model_test = train_and_eval_logreg(c_value, model_label=label)
    logreg_oof_list.append(model_oof)
    logreg_test_list.append(model_test)

logreg_ensemble_oof = np.mean(logreg_oof_list, axis=0)
logreg_ensemble_test = np.mean(logreg_test_list, axis=0)
print(f"\tLR ENSEMBLE: acc={accuracy_score(y_train, logreg_ensemble_oof > 0.5):.4f}, logloss={log_loss(y_train, logreg_ensemble_oof):.4f}\n", flush=True)

"""
2. Train XGBoost models with different parameter sets and average their predictions.
"""
xgb_d4_oof, xgb_d4_test = train_and_eval_xgb(
    xgb_params_depth4, model_label="d4"
)
xgb_d3_oof, xgb_d3_test = train_and_eval_xgb(
    xgb_params_depth3, model_label="d3"
)
xgb_opt_oof, xgb_opt_test = train_and_eval_xgb(
    xgb_params_optuna, model_label="Optuna"
)

xgb_ensemble_oof = (xgb_d4_oof + xgb_d3_oof + xgb_opt_oof) / 3.0
xgb_ensemble_test = (xgb_d4_test + xgb_d3_test + xgb_opt_test) / 3.0
print(f"\tXGB ENSEMBLE: acc={accuracy_score(y_train, xgb_ensemble_oof > 0.5):.4f}, logloss={log_loss(y_train, xgb_ensemble_oof):.4f}\n", flush=True)

"""
3. Train CatBoost models with different parameter sets and average their predictions.
"""
cb_d4_oof, cb_d4_test = train_and_eval_catboost(
    catboost_params_depth4, model_label="d4"
)
cb_d5_oof, cb_d5_test = train_and_eval_catboost(
    catboost_params_depth5, model_label="d5"
)
cb_opt_oof, cb_opt_test = train_and_eval_catboost(
    catboost_params_optuna, model_label="Optuna"
)

catboost_ensemble_oof = (cb_d4_oof + cb_d5_oof + cb_opt_oof) / 3.0
catboost_ensemble_test = (cb_d4_test + cb_d5_test + cb_opt_test) / 3.0
print(f"\tCB ENSEMBLE: acc={accuracy_score(y_train, catboost_ensemble_oof > 0.5):.4f}, logloss={log_loss(y_train, catboost_ensemble_oof):.4f}\n", flush=True)


"""
4. Find optimal blending weights for the 3 model families (LR, XGB, CB) using Scipy minimize.
"""
stacked_oof_matrix = np.column_stack(
    [logreg_ensemble_oof, xgb_ensemble_oof, catboost_ensemble_oof]
)
stacked_test_matrix = np.column_stack(
    [logreg_ensemble_test, xgb_ensemble_test, catboost_ensemble_test]
)

unbiased_blended_oof = np.zeros(len(y_train))
fold_optimal_weights = []

for train_idx, val_idx in cv_folds:
    optimization_result = minimize(
        lambda weights: log_loss(
            y_train[train_idx], stacked_oof_matrix[train_idx] @ weights
        ),
        x0=[0.60, 0.20, 0.20],
        method="SLSQP",
        bounds=[(0, 1)] * 3,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
    )
    best_fold_weights = optimization_result.x
    fold_optimal_weights.append(best_fold_weights)
    unbiased_blended_oof[val_idx] = (
        stacked_oof_matrix[val_idx] @ best_fold_weights
    )

final_blend_weights = np.mean(fold_optimal_weights, axis=0)
final_blended_oof = stacked_oof_matrix @ final_blend_weights
final_blended_test_predictions = stacked_test_matrix @ final_blend_weights

print(f"Optimal weights [LR, XGB, CB]: [{final_blend_weights[0]:.3f}, {final_blend_weights[1]:.3f}, {final_blend_weights[2]:.3f}]")
print(f"BLEND (Unbiased): acc@0.50={accuracy_score(y_train, unbiased_blended_oof > 0.50):.4f}, logloss={log_loss(y_train, unbiased_blended_oof):.4f}")
print(f"BLEND (Final Weighted): acc@0.50={accuracy_score(y_train, final_blended_oof > 0.50):.4f}, logloss={log_loss(y_train, final_blended_oof):.4f}")

#####################

#####################
#9. Submission

sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
Path(f"./output").mkdir(parents=True, exist_ok=True)

DECISION_THRESHOLDS = (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52)

print("-----------------------------------------------------------")

for threshold in DECISION_THRESHOLDS:
    binary_test_predictions = (
        final_blended_test_predictions > threshold
    ).astype(int)

    submission_df = pd.DataFrame({
        "battle_id": test_ids,
        "player_won": binary_test_predictions,
    })

    threshold_str = (
        f"{int(round(threshold * 100)):03d}"
    )
    output_file_path = (
        f"./output/submission_curated_{threshold_str}.csv"
    )
    submission_df.to_csv(output_file_path, index=False)

    final_oof_accuracy = accuracy_score(
        y_train, final_blended_oof > threshold
    )
    unbiased_oof_accuracy = accuracy_score(
        y_train, unbiased_blended_oof > threshold
    )

    win_rate_in_test = submission_df["player_won"].mean()
    total_wins_in_test = submission_df["player_won"].sum()
    total_test_battles = len(submission_df)

    print(f"Threshold {threshold:.2f} saved @ {output_file_path}")
    print(f"OOF acc={final_oof_accuracy:.4f} (unbiased={unbiased_oof_accuracy:.4f})")
    print(f"predicted p1 won in TEST={win_rate_in_test:.4f} ({total_wins_in_test}/{total_test_battles})\n")

print("-----------------------------------------------------------")
