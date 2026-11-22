"""Wing Lab: "Message to the Mainland", a gamified wing-design game.

The game itself (cutscene, controls, flight and rescue animations) runs inside
a Streamlit custom component (app/wing_lab_game/index.html, plain JavaScript).
The component only sends player choices ("launch", "scout", ...) back here.
Everything scientific happens in Python:

REAL (model): each wing's separation_x_over_c comes from the unchanged
src.inference.predict_with_clipping(), via wing_lab_core.
GAME (invented): distance, height, points and upgrades are game rules defined
in wing_lab_core.py and labelled as such on screen.

All state lives in st.session_state under keys prefixed "wl_".
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import get_loaded_model_path, get_model_metrics
import wing_lab_core as game
from wing_lab_component import game_component

GAME_EVENT_KEY = "wl_game_event"
FAMILY_LABEL = {"symmetric": "Symmetric", "cambered": "Cambered", "biomimetic": "🐋 Biomimetic"}

# -----------------------------------------------------------------------------
# State and event handling
# -----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("wl_state", game.new_state())
ss.setdefault("wl_gen", 0)              # bumps on restart so the game frontend resets
ss.setdefault("wl_last_nonce", None)    # each frontend event is processed exactly once
ss.setdefault("wl_quiz_checked", False)


def restart_game() -> None:
    ss.wl_state = game.new_state()
    ss.wl_gen += 1
    ss.wl_quiz_checked = False
    ss.pop("wl_quiz_choice", None)


event = ss.get(GAME_EVENT_KEY)
if isinstance(event, dict) and event.get("nonce") and event["nonce"] != ss.wl_last_nonce:
    ss.wl_last_nonce = event["nonce"]
    if event.get("action") == "restart":
        restart_game()
    else:
        game.handle_event(ss.wl_state, event)

state = ss.wl_state


def frontend_state(s: dict) -> dict:
    """Everything the game frontend needs to draw. JSON-serialisable."""
    unlocks = game.unlocked(s["points"])
    last = None
    if s["rounds"]:
        r = s["rounds"][-1]
        last = {
            "round": r["round"], "points_earned": r["points_earned"], "new_record": r["new_record"],
            "new_unlocks": [{"title": u["title"], "text": u["text"]} for u in r["new_unlocks"]],
            "flights": [{
                "name": f["name"], "family": f["inputs"]["airfoil_family"],
                "amp": f["inputs"]["tubercle_amplitude"], "angle": int(f["inputs"]["angle_of_attack"]),
                "speed": f["inputs"]["airspeed"], "x": f["x_over_c"], "distance": f["distance"],
                "height": f["height"], "reached": f["reached_mainland"],
            } for f in r["flights"]],
        }
    return {
        "gen": ss.wl_gen, "points": s["points"], "best": s["best"], "rounds_flown": len(s["rounds"]),
        "won": s["won"], "intro_seen": s["intro_seen"], "rescue_seen": s["rescue_seen"],
        "unlocks": sorted(unlocks),
        "upgrades": [{k: u[k] for k in ("key", "points", "title", "text")} for u in game.UPGRADES],
        "config": s["config"], "scout": s["scout"], "error": s["error"],
        "starters": [{"name": x["name"], "family": x["airfoil_family"], "angle": x["angle_of_attack"]}
                     for x in game.STARTER_SLOTS],
        "bumps": [{"amp": g["tubercle_amplitude"], "wave": g["tubercle_wavelength"]}
                  for g in game.TUBERCLE_GEOMETRIES],
        "speeds": list(game.SPEEDS),
        "last_round": last,
    }


# -----------------------------------------------------------------------------
# The game
# -----------------------------------------------------------------------------
st.title("🏝️ Wing Lab: Message to the Mainland")
game_component(state=frontend_state(state), key=GAME_EVENT_KEY, default=None)

if state["rounds"]:
    r = state["rounds"][-1]
    st.caption(
        f"Round {r['round']}: "
        + (f"🏆 new record, +{r['points_earned']} points" if r["new_record"] else "no new record, 0 points")
        + f" · **Total: {state['points']} ⭐** · Record: {state['best']:.0f} m"
    )

st.info(
    "**What's real and what's game?** Each wing's **x/c** is a real prediction from WingCheck's "
    "machine-learning model. **Distance, height and points** are invented **game rules**: here, "
    "distance is proportional to x/c (1.5× the x/c flies 1.5× as far). Real wings also depend on "
    "lift, drag, weight and much more. **Weather, time of day and music are just atmosphere**: "
    "the model was never given wind or rain, so they never change any prediction or result.",
    icon="🎮",
)

with st.expander("📖 Field notes: biomimicry and flow separation"):
    st.markdown(
        "**Flow separation.** Air flowing over a wing tries to follow its curved top surface. "
        "When the wing is tilted steeply, the air can't keep up and **breaks away**. Behind that "
        "point the flow is messy and turbulent. WingCheck's model predicts **where** that happens "
        "as a fraction of the chord, **x/c**: 0 = the front edge, 1 = the back edge.\n\n"
        "**Biomimicry.** Humpback whales have bumps called **tubercles** on the front of their "
        "flippers. Wind-tunnel studies on flipper-shaped models found tubercles can change how "
        "airflow separates at high angles, so engineers test them on wings."
    )

with st.expander("🧮 How the game turns x/c into distance and points"):
    st.markdown(
        f"`distance = {game.DISTANCE_SCALE:.0f} m × launch power × x/c` (**game rule, not physics**)\n\n"
        "- **Launch power** (0–1) comes only from launch angle and speed, and can never go above 1.\n"
        f"- So reaching the 1,000 m mainland needs **x/c ≥ {game.WIN_X_OVER_C:.2f}**: only wings the "
        "model predicts keep their flow attached that far back can ever make it.\n"
        "- **Points** are only earned by **beating your own record**: 1 point per 10 m of "
        "improvement. Relaunching the same wing earns nothing, so experiment!"
    )
    if state["rounds"]:
        st.dataframe(pd.DataFrame([{
            "Wing": f["name"], "Type": FAMILY_LABEL[f["inputs"]["airfoil_family"]],
            "Model x/c": round(f["x_over_c"], 3), "Launch power": round(f["launch_power"], 2),
            "Distance (m)": round(f["distance"]),
        } for f in state["rounds"][-1]["flights"]]), hide_index=True, use_container_width=True)
        st.caption(f"The model's typical error is about ±{game.comparison_threshold():.2f} x/c, so "
                   "wings with very close x/c values may not really be different.")

# -----------------------------------------------------------------------------
# Debrief and quiz (after the rescue animation has played)
# -----------------------------------------------------------------------------
QUIZ_OPTIONS = [
    "Yes. Later separation always means a real wing flies farther.",
    "Not necessarily. Real distance also depends on lift, drag, weight and more, and the game made up its distance rule.",
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

if state["won"] and state["rescue_seen"]:
    st.header("🎉 Debrief: Link is home!", divider="green")
    winner = max(state["rounds"][-1]["flights"], key=lambda f: f["distance"])
    w_in = winner["inputs"]
    bump = (f" with {w_in['tubercle_amplitude']:.1f} mm bumps" if w_in["airfoil_family"] == "biomimetic" else "")
    st.markdown(
        f"**{winner['name']}** carried the bottle to the mainland in round {len(state['rounds'])}: a "
        f"{FAMILY_LABEL[w_in['airfoil_family']].lower()} wing{bump}, launched at {w_in['angle_of_attack']:.0f}° "
        f"and {w_in['airspeed']} m/s. The model predicted separation at **x/c = {winner['x_over_c']:.2f}**."
    )
    st.subheader("What was real and what was game?")
    st.table(pd.DataFrame({
        "Real (ML model, trained on CFD)": [
            "Each wing's x/c separation prediction",
            "Which inputs are allowed (only ones the model learned from)",
            "How angle, speed, wing type and bumps changed x/c",
        ],
        "Game (made up for fun)": [
            "Flight distance, height and the animations",
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
if state["rounds"]:
    with st.expander(f"📒 Flight log ({len(state['rounds'])} rounds)"):
        log = game.results_dataframe(state["rounds"])
        st.dataframe(log[["round", "wing", "airfoil_family", "tubercle_amplitude", "angle_of_attack",
                          "airspeed", "model_separation_x_over_c", "game_distance_m",
                          "round_points_earned", "total_points"]],
                     hide_index=True, use_container_width=True)

b1, b2 = st.columns(2)
with b1:
    st.button("🔄 Restart game", on_click=restart_game, use_container_width=True)
with b2:
    st.download_button(
        "⬇️ Download my flight log (CSV)",
        data=game.results_dataframe(state["rounds"]).to_csv(index=False).encode("utf-8") if state["rounds"] else b"",
        file_name="wing_lab_flight_log.csv", mime="text/csv",
        disabled=not state["rounds"], use_container_width=True,
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
        "- The story, animations, distances and points are **conceptual game elements**, not CFD or physics.\n"
        "- This is an educational screening prototype, not a replacement for CFD, wind-tunnel testing, "
        "or professional aerodynamic design."
    )
