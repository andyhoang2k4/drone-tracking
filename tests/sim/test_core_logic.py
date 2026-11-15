import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math

from src.sim.config import SimConfig, TargetConfig, FollowConfig
from src.sim.follow_controller import follow_command
from src.sim.gimbal import GimbalTracker
from src.sim.messages import MessageBoard
from src.sim.search import BearingMemory, search_command
from src.sim.supervisor import FlightSupervisor
from src.sim.target_manager import TargetManager
from src.sim.types import Detection, FlightMode, FrameSize, RCInput, TargetState, Telemetry

F = FrameSize(640, 480)


def det(i, cx, cy=240, h=120, label="person"):
    w = h / 2
    return Detection(i, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), label, 0.9)


def locked_manager(cfg=None, cx=320):
    m = TargetManager(cfg or TargetConfig())
    m.update([det(1, cx), det(2, 100, label="car")], 0.0)
    assert m.select_at(cx, 240) == 1
    assert m.lock_selected()
    return m


def test_select_picks_smallest_containing_box():
    m = TargetManager(TargetConfig())
    big = Detection(1, (0, 0, 400, 400), "car", 0.9)
    small = Detection(2, (100, 100, 200, 200), "person", 0.9)
    m.update([big, small], 0.0)
    assert m.select_at(150, 150) == 2
    assert m.select_at(500, 500) is None


def test_lock_requires_selection():
    m = TargetManager(TargetConfig())
    m.update([det(1, 300)], 0.0)
    assert not m.lock_selected()


def test_state_sequence_lost_reid_failed_follows_claude_md_table():
    cfg = TargetConfig(lost_timeout_s=1.0, reid_timeout_s=2.0)
    m = locked_manager(cfg)
    assert m.update([det(1, 320)], 0.1).state == TargetState.TRACKING
    e = m.update([], 0.5)
    assert e.state == TargetState.LOST and e.bbox is not None and 0 < e.prediction_confidence < 1
    e = m.update([], 1.5)
    assert e.state == TargetState.REIDENTIFYING and e.bbox is None
    e = m.update([], 3.5)
    assert e.state == TargetState.FAILED


def test_reacquire_same_label_near_prediction_with_new_id():
    m = locked_manager(TargetConfig(lost_timeout_s=1.0, reid_timeout_s=5.0))
    for k in range(1, 6):  # muc tieu di sang phai 100 px/s
        m.update([det(1, 320 + 100 * 0.1 * k)], 0.1 * k)
    m.update([], 1.0)
    e = m.update([det(9, 330 + 100 * 0.6)], 1.1)
    assert e.state == TargetState.TRACKING and m.locked_id == 9


def test_no_reacquire_for_other_label_or_far_away():
    m = locked_manager(TargetConfig(lost_timeout_s=1.0, reid_timeout_s=5.0))
    assert m.update([det(7, 320, label="car")], 0.5).state == TargetState.LOST
    assert m.update([det(8, 600)], 0.6).state == TargetState.LOST


def test_unlock_clears_estimate():
    m = locked_manager()
    m.unlock()
    assert m.update([det(1, 320)], 0.2) is None


def test_follow_signs():
    cfg = FollowConfig()
    right = follow_command((400, 180, 460, 300), F, 100, cfg)
    left = follow_command((100, 180, 160, 300), F, 100, cfg)
    assert right.yaw_rate > 0 > left.yaw_rate
    far = follow_command((300, 220, 340, 260), F, 100, cfg)  # nho = xa -> tien
    near = follow_command((200, 40, 440, 440), F, 100, cfg)  # to = gan -> lui
    assert far.forward > 0 > near.forward
    assert near.forward >= -cfg.max_back_ms and far.forward <= cfg.max_fwd_ms
    centred = follow_command((300, 180, 340, 300), F, 100, cfg)
    assert centred.yaw_rate == 0.0


def test_follow_does_not_advance_when_box_cut_at_bottom_edge():
    cfg = FollowConfig()
    free = follow_command((300, 300, 330, 340), F, 100, cfg)  # nho, khong cham mep -> tien
    cut = follow_command((300, 440, 330, 480), F, 100, cfg)  # cung kich thuoc nhung cham mep duoi (F cao 480)
    assert free.forward > 0 and cut.forward == 0.0


def test_follow_slows_forward_when_target_far_off_axis():
    cfg = FollowConfig()
    on = follow_command((300, 220, 340, 260), F, 100, cfg)
    off = follow_command((600, 220, 640, 260), F, 100, cfg)
    assert off.forward < on.forward


def test_gimbal_moves_toward_target_and_clamps():
    from src.sim.config import GimbalConfig
    g = GimbalTracker(GimbalConfig())
    p0, y0 = g.pitch_deg, g.yaw_deg
    p, y = g.update((500, 40, 540, 100), F, 100)  # tren-phai
    assert y > y0 and p > p0
    for _ in range(200):
        p, y = g.update((630, 0, 640, 10), F, 100)
    assert y <= GimbalConfig().yaw_limit_deg and p <= GimbalConfig().pitch_max_deg
    g.reset()
    assert g.yaw_deg == 0.0


def test_bearing_memory_rate_and_wrap():
    bm = BearingMemory(1.0)
    for k in range(10):
        bm.add(k * 0.1, math.radians(170 + 5 * k))  # vuot qua 180 do
    last, rate = bm.last_and_rate()
    assert abs(rate - math.radians(50)) < 0.05


def test_search_turns_toward_last_bearing():
    from src.sim.config import SearchConfig
    v = search_command(math.radians(40), 0.0, 1.0, 0.0, 2.0, SearchConfig())
    assert v.yaw_rate > 0 and v.forward > 0
    v2 = search_command(math.radians(-40), 0.0, 1.0, 0.0, 2.0, SearchConfig())
    assert v2.yaw_rate < 0


# ---- supervisor ----
def make_sup(**kw):
    cfg = SimConfig()
    cfg.target = TargetConfig(lost_timeout_s=1.0, reid_timeout_s=3.0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    board = MessageBoard()
    return cfg, board, FlightSupervisor(cfg, board)


def texts(board, now):
    return [m.text for m in board.active(now)]


def test_auto_without_target_stays_manual_and_prompts_with_candidates():
    cfg, board, sup = make_sup()
    m = TargetManager(cfg.target)
    est = m.update([det(3, 200, label="person"), det(5, 400, label="car")], 0.0)
    out = sup.step(0.0, RCInput(mode_switch=FlightMode.AUTO_FOLLOW), est, m.candidates, F, Telemetry())
    assert out.desired_mode == FlightMode.STABILIZED and not out.auto_engaged
    msg = " ".join(texts(board, 0.0))
    assert "CHUA CO MUC TIEU" in msg and "person#3" in msg and "car#5" in msg


def test_lock_then_switch_auto_engages_and_follows():
    cfg, board, sup = make_sup()
    m = locked_manager(cfg.target, cx=450)
    est = m.update([det(1, 450)], 0.1)
    out = sup.step(0.1, RCInput(mode_switch=FlightMode.AUTO_FOLLOW), est, m.candidates, F, Telemetry())
    assert out.auto_engaged and out.desired_mode == FlightMode.AUTO_FOLLOW
    assert out.velocity.yaw_rate > 0


def test_lock_while_switch_in_auto_engages_when_enabled_and_not_when_disabled():
    for flag, expect in ((True, True), (False, False)):
        cfg, board, sup = make_sup(auto_engage_on_lock=flag)
        m = TargetManager(cfg.target)
        est = m.update([det(1, 320)], 0.0)
        rc = RCInput(mode_switch=FlightMode.AUTO_FOLLOW)
        sup.step(0.0, rc, est, m.candidates, F, Telemetry())
        m.select_at(320, 240); m.lock_selected()
        est = m.update([det(1, 320)], 0.1)
        out = sup.step(0.1, rc, est, m.candidates, F, Telemetry())
        assert out.auto_engaged is expect


def test_lost_then_search_then_failed_hover_and_messages():
    cfg, board, sup = make_sup()
    m = locked_manager(cfg.target, cx=320)
    rc = RCInput(mode_switch=FlightMode.AUTO_FOLLOW)
    tel = Telemetry(yaw_rad=0.0)
    t = 0.0
    for k in range(10):  # muc tieu di ve ben phai
        t = 0.1 * k
        est = m.update([det(1, 320 + 20 * k)], t)
        sup.step(t, rc, est, m.candidates, F, tel)
    t += 0.5
    est = m.update([], t)
    out = sup.step(t, rc, est, [], F, tel)
    assert est.state == TargetState.LOST and out.status == "AUTO:LOST"
    assert "MUC TIEU BIEN MAT" in texts(board, t)
    t += 1.0
    est = m.update([], t)
    out = sup.step(t, rc, est, [], F, tel)
    assert est.state == TargetState.REIDENTIFYING and out.status == "AUTO:REIDENTIFYING"
    assert out.velocity.forward > 0 and out.velocity.yaw_rate >= 0
    t += 5.0
    est = m.update([], t)
    out = sup.step(t, rc, est, [], F, tel)
    assert est.state == TargetState.FAILED and out.velocity.forward == 0 and out.velocity.yaw_rate == 0


def test_switch_back_to_manual_releases_auto():
    cfg, board, sup = make_sup()
    m = locked_manager(cfg.target)
    est = m.update([det(1, 320)], 0.1)
    sup.step(0.1, RCInput(mode_switch=FlightMode.AUTO_FOLLOW), est, m.candidates, F, Telemetry())
    out = sup.step(0.2, RCInput(mode_switch=FlightMode.ACRO), est, m.candidates, F, Telemetry())
    assert out.desired_mode == FlightMode.ACRO and not out.auto_engaged


def test_arm_edges_and_manual_passthrough():
    cfg, board, sup = make_sup()
    rc = RCInput(roll=0.3, pitch=-0.2, yaw=0.1, throttle=0.4)
    assert sup.step(0.0, rc, None, [], F, Telemetry()).arm is None
    rc.arm_switch = True
    out = sup.step(0.1, rc, None, [], F, Telemetry())
    assert out.arm is True and out.manual == (-0.2, 0.3, 0.4, 0.1)
    assert sup.step(0.2, rc, None, [], F, Telemetry()).arm is None
    rc.arm_switch = False
    assert sup.step(0.3, rc, None, [], F, Telemetry()).arm is False


def test_gimbal_follows_locked_target_in_manual_mode_and_resets_on_unlock():
    cfg, board, sup = make_sup()
    m = locked_manager(cfg.target, cx=500)
    est = m.update([det(1, 500)], 0.1)
    out = sup.step(0.1, RCInput(mode_switch=FlightMode.STABILIZED), est, m.candidates, F, Telemetry())
    assert not out.auto_engaged and out.gimbal[1] > 0
    m.unlock()
    out = sup.step(0.2, RCInput(), None, [], F, Telemetry())
    assert out.gimbal == (cfg.gimbal.default_pitch_deg, 0.0)


def test_failed_is_terminal_no_auto_reacquire():
    cfg = TargetConfig(lost_timeout_s=1.0, reid_timeout_s=2.0)
    m = locked_manager(cfg)
    m.update([], 0.5)
    assert m.update([], 4.0).state == TargetState.FAILED
    assert m.update([det(9, 320)], 4.1).state == TargetState.FAILED  # ID moi giong het nhung khong tu nhan
    assert m.update([det(1, 320)], 4.2).state == TargetState.TRACKING  # ID goc quay lai thi van nhan


def test_reid_phase_ambiguous_candidates_not_adopted_but_unique_is():
    cfg = TargetConfig(lost_timeout_s=1.0, reid_timeout_s=5.0)
    m = locked_manager(cfg)
    m.update([], 0.5)
    amb = m.update([det(8, 330), det(9, 360)], 1.5)  # 2 nguoi cung nhan lop -> mo ho
    assert amb.state == TargetState.REIDENTIFYING
    ok = m.update([det(9, 360), det(5, 100, label="car")], 1.6)  # chi con 1 'person'
    assert ok.state == TargetState.TRACKING and m.locked_id == 9


def test_reacquire_radius_is_capped():
    cfg = TargetConfig(lost_timeout_s=1.0, reid_timeout_s=50.0, reacquire_max_px=100.0)
    m = locked_manager(cfg, cx=100)
    m.update([], 0.5)
    assert m.update([det(9, 600)], 20.0).state == TargetState.REIDENTIFYING
