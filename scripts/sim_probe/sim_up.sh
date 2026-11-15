#!/usr/bin/env bash
# Khoi dong Gazebo server (headless) + PX4 SITL standalone gan vao x500_oracle_0 cua world follow_v2.
# Chi chay tren Ubuntu/cloud (CLAUDE.md muc 3). Bien: SIM_DIR (mac dinh ~/sim_probe/follow_v2), PX4_DIR, TAKEOFF_ALT (mac dinh 12 m).
# Lenh console PX4: echo "<lenh>" >> $SIM_DIR/px4_cmds.txt
set -u
SIM_DIR=${SIM_DIR:-$HOME/sim_probe/follow_v2}
PX4_DIR=${PX4_DIR:-$HOME/PX4-Autopilot}
TAKEOFF_ALT=${TAKEOFF_ALT:-12}  # m; cao hon cay/nha trong follow_v2 (lop sim khong ne vat can)
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/sim_down.sh" >/dev/null 2>&1
cd "$SIM_DIR" || exit 1
unset VIRTUAL_ENV
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:-} GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}  # gz_env.sh dung cac bien nay (set -u)
. "$PX4_DIR/build/px4_sitl_default/gz_env.sh"
export GZ_SIM_RESOURCE_PATH=$SIM_DIR/models:$HOME/sim_probe/fuel_links:$GZ_SIM_RESOURCE_PATH
export GZ_IP=127.0.0.1
(unset DISPLAY; nohup nice -n 5 gz sim -v 3 -s -r --headless-rendering follow_v2.sdf > gz.log 2>&1 &)
echo "Cho Gazebo..."; sleep 15
: > px4_cmds.txt
(cd "$PX4_DIR/build/px4_sitl_default/rootfs" && nohup env PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 \
  PX4_GZ_MODEL_NAME=x500_oracle_0 PX4_GZ_WORLD=follow_v2 \
  bash -c "tail -f $SIM_DIR/px4_cmds.txt | ../bin/px4" > "$SIM_DIR/px4.log" 2>&1 &)
echo "Cho PX4..."; sleep 25
# SITL khong co GCS: NAV_DLL_ACT=0 de arm duoc (heartbeat lop sim la onboard controller).
printf 'param set NAV_DLL_ACT 0\nparam set MIS_TAKEOFF_ALT %s\n' "$TAKEOFF_ALT" >> px4_cmds.txt
sleep 3
grep -aq "gz_bridge\] world: follow_v2" px4.log && echo "OK: PX4 da gan vao drone" || { echo "LOI: xem $SIM_DIR/px4.log, gz.log"; exit 1; }
