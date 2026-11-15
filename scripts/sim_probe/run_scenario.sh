#!/usr/bin/env bash
# Mot lan chay vong kin tu dau tren world follow_v2: sim_up -> cat canh TAKEOFF_ALT (mac dinh 12 m) -> run_closed_loop.py -> sim_down.
# Chi chay tren Ubuntu/cloud. Tham so sau "--" chuyen thang cho run_closed_loop.py (vd --detector yolo --weights W).
#   SIM_DIR=~/sim_probe/follow_v2 ./scripts/sim_probe/run_scenario.sh --out ~/sim_probe/follow_v2/runs/r1 --detector oracle
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SIM_DIR=${SIM_DIR:-$HOME/sim_probe/follow_v2}
PY=${PY:-python3}
export TAKEOFF_ALT=${TAKEOFF_ALT:-12}
"$HERE/sim_up.sh" || exit 1
echo "commander takeoff" >> "$SIM_DIR/px4_cmds.txt"
export GZ_IP=127.0.0.1
z=0
for i in $(seq 1 60); do
  sleep 1
  z=$(gz topic -e -n 1 -t /world/follow_v2/pose/info 2>/dev/null | grep -A6 'name: "x500_oracle_0"' | grep -E ' z:' | awk '{print $2}')
  awk -v z="$z" -v a="$TAKEOFF_ALT" 'BEGIN{exit !(z>a-0.4)}' && break
done
echo "do cao truoc khi chay: $z"
sleep 3
"$PY" "$HERE/run_closed_loop.py" --scenario "$SIM_DIR/scenario.json" "$@"
rc=$?
"$HERE/sim_down.sh"
exit $rc
