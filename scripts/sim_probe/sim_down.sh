#!/usr/bin/env bash
# Tat Gazebo server + PX4 SITL + xe cua world follow_v2 (mau neo dau dong).
SIM_DIR=${SIM_DIR:-$HOME/sim_probe/follow_v2}
PATS=("^gz sim -v 3 -s -r --headless-rendering follow_v2.sdf" "^\.\./bin/px4" "^bash -c tail -f $SIM_DIR/px4_cmds.txt" "^tail -f $SIM_DIR/px4_cmds.txt" "^[^ ]*python[^ ]* [^ ]*drive_vehicles\.py")
for p in "${PATS[@]}"; do pkill -TERM -f "$p"; done
sleep 3
for p in "${PATS[@]}"; do pkill -KILL -f "$p"; done
true
