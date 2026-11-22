"""Wing Lab: "Message to the Mainland", a gamified wing-design game.

REAL (model): each wing's separation_x_over_c comes from the unchanged
src.inference.predict_with_clipping(), via wing_lab_core.fly().
GAME (invented): distance, height, points and upgrades are game rules defined
in wing_lab_core.py and labelled as such on screen.

All game state lives in st.session_state under keys prefixed "wl_".
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.inference import get_loaded_model_path, get_model_metrics
import wing_lab_core as game

SLOT_COLOURS = ["#ff8c42", "#e84a5f", "#8e44ad"]
FAMILY_LABEL = {"symmetric": "Symmetric", "cambered": "Cambered", "biomimetic": "🐋 Biomimetic"}
FLIGHT_HTML = (Path(__file__).parent / "wing_lab_flight.html").read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("wl_points", 0)
ss.setdefault("wl_rounds", [])          # [{round, flights, points, new_unlocks, airspeed}]
ss.setdefault("wl_best", 0.0)
ss.setdefault("wl_won", False)
ss.setdefault("wl_error", None)
ss.setdefault("wl_quiz_checked", False)
# Saved wing-type choices. Kept apart from the dropdown widgets: Streamlit's
# selectbox ignores a value pre-set in session state, and saved values also
# survive switching pages.
ss.setdefault("wl_families", [s["airfoil_family"] for s in game.STARTER_SLOTS])
for i in range(len(game.STARTER_SLOTS)):
    ss.setdefault(f"wl_bump_{i}", game.DEFAULT_BUMP_INDEX)
ss.setdefault("wl_angle", 8)
ss.setdefault("wl_speed", game.DEFAULT_SPEED)

unlocks = game.unlocked(ss.wl_points)


def restart_game() -> None:
    for key in [k for k in ss.keys() if str(k).startswith("wl_")]:
        del ss[key]


def save_family(i: int) -> None:
    families = list(ss.wl_families)
    families[i] = ss[f"wl_family_{i}"]
    ss.wl_families = families


def current_slots(unl: set) -> list:
    """Wing settings the player can currently use. Locked settings stay at game defaults."""
    slots = []
    for i, starter in enumerate(game.STARTER_SLOTS):
        family = ss.wl_families[i] if "wing_types" in unl else starter["airfoil_family"]
        if family == "biomimetic" and "biomimetic" not in unl:
            family = starter["airfoil_family"]
        bump = ss[f"wl_bump_{i}"] if "bump_height" in unl else game.DEFAULT_BUMP_INDEX
        angle = ss.wl_angle if "angle" in unl else starter["angle_of_attack"]
        slots.append({"name": f"Wing {'ABC'[i]}", "airfoil_family": family,
                      "angle_of_attack": int(angle), "bump_index": bump})
    return slots


def current_speed(unl: set) -> int:
    return ss.wl_speed if "speed" in unl else game.DEFAULT_SPEED


def launch() -> None:
    unl = game.unlocked(ss.wl_points)
    try:
        speed = current_speed(unl)
        flights = game.fly(current_slots(unl), speed, unl)
    except Exception as exc:
        ss.wl_error = str(exc)
        return
    ss.wl_error = None
    pts = game.round_points(flights, ss.wl_best)
    before = game.unlocked(ss.wl_points)
    ss.wl_points += pts["total"]
    ss.wl_best = pts["best"]
    new = [u for u in game.UPGRADES if u["key"] in game.unlocked(ss.wl_points) - before]
    ss.wl_rounds.append({"round": len(ss.wl_rounds) + 1, "flights": flights, "points": pts,
                         "new_unlocks": new, "airspeed": speed})
    if any(f["reached_mainland"] for f in flights):
        ss.wl_won = True


# -----------------------------------------------------------------------------
# Header and story
# -----------------------------------------------------------------------------
st.title("🏝️ Wing Lab: Message to the Mainland")

if not ss.wl_rounds:
    st.markdown(
        "**Kai is stranded on a tiny island.** The mainland is **1,000 m** across the ocean. "
        "Kai has written a rescue message and needs a glider that can carry it all the way there.\n\n"
        "You'll launch **three wings per round**. Every metre they fly earns points, and points "
        "unlock upgrades for Kai's workshop. Get one wing to the mainland to save Kai!"
    )
    with st.expander("📖 Field notes: biomimicry and flow separation", expanded=True):
        st.markdown(
            "**Flow separation.** Air flowing over a wing tries to follow its curved top surface. "
            "When the wing is tilted steeply, the air can't keep up and **breaks away**. Behind that "
            "point the flow is messy and turbulent. WingCheck's model predicts **where** that "
            "happens as a fraction of the chord, **x/c**: 0 = the front edge, 1 = the back edge.\n\n"
            "**Biomimicry.** Humpback whales have bumps called **tubercles** on the front of their "
            "flippers. Wind-tunnel studies on flipper-shaped models found tubercles can change how "
            "airflow separates at high angles, so engineers test them on wings. Keep playing to unlock one!"
        )

st.info(
    "**What's real and what's game?** Each wing's **x/c** is a real prediction from WingCheck's "
    "machine-learning model. The **flight distance, height and points** are invented **game rules**: "
    "in this game a higher x/c makes a wing glide higher and farther. Real wings also depend on lift, "
    "drag, weight and much more.",
    icon="🎮",
)

# -----------------------------------------------------------------------------
# Scoreboard and upgrades
# -----------------------------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("⭐ Points", ss.wl_points)
m2.metric("🚀 Rounds flown", len(ss.wl_rounds))
m3.metric("📏 Best flight", f"{ss.wl_best:.0f} m", help="Game distance. The mainland is 1,000 m away.")

nxt = game.next_upgrade(ss.wl_points)
if nxt:
    prev = max([0] + [u["points"] for u in game.UPGRADES if u["points"] <= ss.wl_points])
    st.progress((ss.wl_points - prev) / (nxt["points"] - prev),
                text=f"Next upgrade: {nxt['title']} at {nxt['points']} points")
else:
    st.progress(1.0, text="All upgrades unlocked!")

with st.expander(f"🧰 Workshop upgrades ({len(unlocks)}/{len(game.UPGRADES)} unlocked)"):
    for u in game.UPGRADES:
        mark = "✅" if u["key"] in unlocks else f"🔒 {u['points']} pts"
        st.markdown(f"{mark} **{u['title']}**: {u['text']}")

# -----------------------------------------------------------------------------
# Workshop: configure the three wings
# -----------------------------------------------------------------------------
if not ss.wl_won:
    st.header("🛠️ Kai's workshop", divider="blue")
    if not unlocks:
        st.caption("Kai only has scrap materials for now, so the game builds your first three wings. "
                   "Launch them to earn points!")

    family_options = ["symmetric", "cambered"] + (["biomimetic"] if "biomimetic" in unlocks else [])
    if "angle" in unlocks:
        st.slider("📐 Launch angle (angle of attack) for all three wings", 0, 24, key="wl_angle",
                  format="%d°", help="All wings in a round share the angle, so it's a fair comparison. "
                  "In the game a steeper launch gives more launch power, but the model may predict earlier "
                  "separation (lower x/c).")
    if "speed" in unlocks:
        st.radio("💨 Launch speed", game.SPEEDS, key="wl_speed", horizontal=True,
                 format_func=lambda v: f"{v} m/s", help="30 m/s gives more launch power in the game. "
                 "The model decides what happens to x/c.")

    slots = current_slots(unlocks)
    speed = current_speed(unlocks)
    cols = st.columns(3)
    for i, (col, slot) in enumerate(zip(cols, slots)):
        with col, st.container(border=True):
            st.markdown(f"<span style='color:{SLOT_COLOURS[i]};font-size:1.4rem'>▶</span> "
                        f"**{slot['name']}** · {game.STARTER_SLOTS[i]['name']}", unsafe_allow_html=True)
            if "wing_types" in unlocks:
                st.selectbox("Wing type", family_options, key=f"wl_family_{i}",
                             index=family_options.index(slot["airfoil_family"]),
                             format_func=FAMILY_LABEL.get, on_change=save_family, args=(i,))
            else:
                st.markdown(f"Wing type: **{FAMILY_LABEL[slot['airfoil_family']]}** 🔒")
            if slot["airfoil_family"] == "biomimetic":
                if "bump_height" in unlocks:
                    st.select_slider("Bump height", options=list(range(len(game.TUBERCLE_GEOMETRIES))),
                                     key=f"wl_bump_{i}",
                                     format_func=lambda b: f"{game.TUBERCLE_GEOMETRIES[b]['tubercle_amplitude']:.1f} mm")
                g = game.TUBERCLE_GEOMETRIES[slot["bump_index"]]
                st.caption(f"Bumps: height {g['tubercle_amplitude']:.1f} mm, spacing "
                           f"{g['tubercle_wavelength']:.1f} mm"
                           + ("" if "bump_height" in unlocks else " 🔒"))
            st.caption(f"Angle {slot['angle_of_attack']}° · {speed} m/s"
                       + ("" if "angle" in unlocks else " 🔒"))
            if "wind_scout" in unlocks:
                try:
                    st.markdown(f"🔭 Scout: model predicts **x/c = {game.predict_x(slot, speed)['x_over_c']:.2f}**")
                except Exception as exc:
                    st.caption(f"Scout unavailable: {exc}")

    if "bump_height" in unlocks:
        st.caption("In the CFD data, bump height and spacing always changed together, so picking a "
                   "height also sets the spacing. The model only knows these six real designs.")

    st.button("🚀 Launch!", type="primary", on_click=launch, use_container_width=True)
    if ss.wl_error:
        st.error(f"Launch failed: {ss.wl_error}")

# -----------------------------------------------------------------------------
# Latest round: animation and results
# -----------------------------------------------------------------------------
if ss.wl_rounds:
    rnd = ss.wl_rounds[-1]
    flights = rnd["flights"]
    st.header(f"📡 Round {rnd['round']} results", divider="blue")

    anim = [{"name": f["name"], "distance": f["distance"], "height": f["height"],
             "x_over_c": f["x_over_c"], "reached": f["reached_mainland"], "colour": SLOT_COLOURS[i]}
            for i, f in enumerate(flights)]
    html = (FLIGHT_HTML.replace("__FLIGHTS__", json.dumps(anim))
            .replace("__MAINLAND__", str(game.MAINLAND_DISTANCE))
            + f"<!-- round {rnd['round']} -->")   # new round = new iframe = animation replays
    components.html(html, height=350)

    cols = st.columns(3)
    for i, (col, f) in enumerate(zip(cols, flights)):
        with col, st.container(border=True):
            st.markdown(f"<span style='color:{SLOT_COLOURS[i]};font-size:1.4rem'>▶</span> **{f['name']}** · "
                        f"{FAMILY_LABEL[f['inputs']['airfoil_family']]}", unsafe_allow_html=True)
            st.metric("🎮 Game distance", f"{f['distance']:.0f} m" + (" 🏁" if f["reached_mainland"] else ""))
            st.metric("🔬 Model x/c (real prediction)", f"{f['x_over_c']:.2f}")
            st.caption(f"Air stays attached over about the first {f['x_over_c'] * 100:.0f}% of the chord.")
            if f["was_clipped"]:
                st.warning(f"Raw model output {f['raw']:.3f} was outside 0–1 and clipped. Unreliable.")

    p = rnd["points"]
    st.success(f"⭐ +{p['base']} points for distance flown"
               + (f" · +{p['bonus']} new personal best!" if p["bonus"] else "")
               + f" → **{ss.wl_points} total**")
    for u in rnd["new_unlocks"]:
        st.toast(f"Unlocked {u['title']}!", icon="🔓")
        st.info(f"🔓 **New upgrade: {u['title']}**. {u['text']}")

    with st.expander("🧮 How the game turned x/c into distance"):
        st.markdown(
            "`distance = 1000 m × launch power × glide efficiency` (**game rule, not physics**)\n\n"
            "- **Launch power** (0–1) comes only from angle and speed, and can never go above 1, "
            "so a powerful launch alone can't cross the ocean.\n"
            f"- **Glide efficiency** comes from the model's x/c. It equals 1 at x/c = {game.WIN_X_OVER_C:.2f}, "
            f"so **only wings with x/c ≥ {game.WIN_X_OVER_C:.2f} can ever reach the mainland**."
        )
        st.dataframe(pd.DataFrame([{
            "Wing": f["name"], "Model x/c": round(f["x_over_c"], 3),
            "Launch power": round(f["launch_power"], 2), "Glide efficiency": round(f["glide_efficiency"], 2),
            "Distance (m)": round(f["distance"]),
        } for f in flights]), hide_index=True, use_container_width=True)
        st.caption(f"The model's typical error is about ±{game.comparison_threshold():.2f} x/c, so "
                   "wings with very close x/c values may not really be different.")

# -----------------------------------------------------------------------------
# Win screen, debrief, quiz
# -----------------------------------------------------------------------------
QUIZ_OPTIONS = [
    "Yes. Later separation always means a real wing flies farther.",
    "Not necessarily. In real life distance also depends on lift, drag, weight and more, and the game made up its distance rule.",
    "No. Later separation always means more drag, so real wings fly shorter.",
]
QUIZ_CORRECT = 1
QUIZ_FEEDBACK = [
    "Not quite. In this game we *chose* to make higher x/c fly farther. Real flight distance also "
    "depends on lift, drag, weight and conditions the model doesn't predict.",
    "Correct! The model predicts one thing, where flow separates. The game invented the link to "
    "distance. Real engineers would also check lift, drag and weight, and confirm with CFD or a wind tunnel.",
    "Not quite. Later separation doesn't *always* mean more drag. The real point is that this model "
    "predicts separation location only, so it can't tell you about drag or distance at all.",
]

if ss.wl_won:
    if not ss.get("wl_celebrated"):   # celebrate once, not on every rerun
        st.balloons()
        ss.wl_celebrated = True
    st.header("🎉 Kai is rescued!", divider="green")
    winner = max(ss.wl_rounds[-1]["flights"], key=lambda f: f["distance"])
    w_in = winner["inputs"]
    st.markdown(
        f"**{winner['name']}** carried the message to the mainland in round {len(ss.wl_rounds)} with a "
        f"{FAMILY_LABEL[w_in['airfoil_family']].lower()} wing at {w_in['angle_of_attack']:.0f}° and "
        f"{w_in['airspeed']} m/s. The model predicted separation at **x/c = {winner['x_over_c']:.2f}**."
    )
    st.subheader("What was real and what was game?")
    st.table(pd.DataFrame({
        "Real (ML model, trained on CFD)": [
            "Each wing's x/c separation prediction",
            "Which inputs are allowed (only ones the model learned from)",
            "How angle, speed, wing type and bumps changed x/c",
        ],
        "Game (made up for fun)": [
            "Flight distance and height",
            "Launch power from angle and speed",
            "Points, upgrades, and the mainland at 1,000 m",
        ],
    }))
    st.subheader("🧠 Think like an engineer")
    choice = st.radio(
        "In the game, wings with higher x/c flew farther. Does later separation mean a real wing always flies farther?",
        options=range(len(QUIZ_OPTIONS)), format_func=lambda i: QUIZ_OPTIONS[i], index=None,
        key="wl_quiz_choice", on_change=lambda: ss.update(wl_quiz_checked=False),
    )
    if st.button("Check my answer", disabled=choice is None):
        ss.wl_quiz_checked = True
    if ss.wl_quiz_checked and choice is not None:
        (st.success if choice == QUIZ_CORRECT else st.info)(QUIZ_FEEDBACK[choice])

# -----------------------------------------------------------------------------
# Flight log, restart, download, model limits
# -----------------------------------------------------------------------------
if ss.wl_rounds:
    with st.expander(f"📒 Flight log ({len(ss.wl_rounds)} rounds)"):
        log = game.results_dataframe(ss.wl_rounds)
        st.dataframe(log[["round", "wing", "airfoil_family", "tubercle_amplitude", "angle_of_attack",
                          "airspeed", "model_separation_x_over_c", "game_distance_m", "reached_mainland"]],
                     hide_index=True, use_container_width=True)

st.divider()
b1, b2 = st.columns(2)
with b1:
    st.button("🔄 Restart game", on_click=restart_game, use_container_width=True)
with b2:
    st.download_button(
        "⬇️ Download my flight log (CSV)",
        data=game.results_dataframe(ss.wl_rounds).to_csv(index=False).encode("utf-8") if ss.wl_rounds else b"",
        file_name="wing_lab_flight_log.csv", mime="text/csv",
        disabled=not ss.wl_rounds, use_container_width=True,
    )

with st.expander("ℹ️ About the model and its limits"):
    metrics = get_model_metrics()
    mae = f"{metrics['mae']:.3f} x/c" if "mae" in metrics else "N/A"
    r2 = f"{metrics['r2']:.3f}" if "r2" in metrics else "N/A"
    st.markdown(
        f"- **Model file:** `{Path(get_loaded_model_path()).name}` (Extra Trees regressor)\n"
        "- **Trained on:** about 1,700 OpenFOAM CFD simulations (Phase 2 dataset)\n"
        f"- **Held-out test error:** MAE {mae}, R² {r2}\n"
        "- **Predicts:** separation location `separation_x_over_c` only\n"
        "- **Does not predict:** lift, drag, flight distance, efficiency, battery life, or emissions\n"
        "- **Inputs allowed in this game:** symmetric, cambered or biomimetic wings; the six simulated "
        "tubercle designs; angles 0–24° in whole degrees; airspeeds 15 or 30 m/s\n"
        "- The flight animation and distances are **conceptual game elements**, not CFD or physics.\n"
        "- This is an educational screening prototype, not a replacement for CFD, wind-tunnel testing, "
        "or professional aerodynamic design."
    )
