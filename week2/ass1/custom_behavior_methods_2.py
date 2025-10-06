
import numpy as np
from math import atan2, pi
from irsim.lib import register_behavior


_GAMMA = 0.99     # used in (I - γP)V = R

# policy π parameters
_V0 = 0.6
_KW = 1.2
_KR = 0.6



BIN_D, BIN_TH = 10, 12    # Number of distance bins and angle bins
D_MAX = 3.0               # Maximum considered distance

# ADP statistics
_S = BIN_D * BIN_TH
_trans = np.zeros((_S, _S), dtype=np.float32)   # N(s,s')
_vis   = np.zeros((_S,), dtype=np.int32)        # N(s)
_Rsum  = np.zeros((_S,), dtype=np.float32)      # ∑r(s)
_V     = np.zeros((_S,), dtype=np.float32)      # estimated V(s)
_last_state = None


_ADP_EVERY = 200
_step = 0

def passive_save(path="circle follow.npz"):
    np.savez(path, V=_V, trans=_trans, vis=_vis, Rsum=_Rsum,
             bins_d=BIN_D, bins_th=BIN_TH, dmax=D_MAX)

# ==== tools ====
def _wrap(a):  # wrap to [-pi,pi]
    return (a + pi) % (2 * pi) - pi

def _ensure_circle_params(ego,**kwargs):
    #  default value: center[5.0, 5.0],radius 1.0, changed through yaml file
    ego.circle_center = np.array([5.0, 5.0], dtype=np.float32)
    ego.circle_radius = 1.0
    if "center" in kwargs:
        ego.circle_center = np.asarray(kwargs["center"], dtype=np.float32).reshape(2)
    if "radius" in kwargs:
        ego.circle_radius = float(kwargs["radius"])
    return ego.circle_center, ego.circle_radius


def _state_features(ego,**kwargs):
    s = np.asarray(ego.state, np.float32).reshape(-1)
    x, y, th = float(s[0]), float(s[1]), float(s[2])
    _C, _R = _ensure_circle_params(ego, **kwargs)
    dx, dy = x - _C[0], y - _C[1]     # distance  to the center of the circle in x,y direction
    phi_tan = atan2(dy, dx) + pi/2    # change to tangent direction
    dth = _wrap(th - phi_tan)         # current direction- tangent direction
    r=np.hypot(dx, dy)            # distance  to the center of the circle
    e_r = r - _R  # difference between distance and R
    dist = abs(e_r)
    return dist, dth, e_r

def _to_index(dist, dth):   # convert continuous variables into a single discrete state index
    di = int(np.clip(dist / D_MAX * BIN_D, 0, BIN_D-1))  # convert to integers in [0,BIN_D-1]
    ti = int(np.floor((dth + pi) / (2*pi) * BIN_TH))
    ti = int(np.clip(ti, 0, BIN_TH-1))    # convert to integers in [0,BIN_TH-1]
    return di * BIN_TH + ti

def _adp_policy_evaluation():
    #  (I - γP)V = R
    S_vis = np.where(_vis > 0)[0]
    if S_vis.size == 0:
        return
    P = np.zeros((S_vis.size, S_vis.size), dtype=np.float32)
    R = np.zeros((S_vis.size,), dtype=np.float32)
    idx_map = {s:i for i,s in enumerate(S_vis)}
    # Naturalization: P , R
    for si in S_vis:
        i = idx_map[si]
        row_sum = float(_trans[si].sum()) # sum of transitions from state si to all other states
        if row_sum > 0:
            for sj in S_vis:
                j = idx_map[sj]
                P[i, j] = _trans[si, sj] / row_sum # probability of state transitions approximated by previous counts
        else:
            P[i, i] = 1.0
        R[i] = _Rsum[si] / max(_vis[si], 1)
    A = np.eye(S_vis.size, dtype=np.float32) - _GAMMA * P
    try:
        V_sub = np.linalg.solve(A, R)
    except np.linalg.LinAlgError:
        V_sub = np.linalg.lstsq(A, R, rcond=None)[0]     #  solve the least squares problem
    _V[S_vis] = V_sub

def _shape_like_ref(u, ref_like):
    arr = np.asarray(u, np.float32).reshape(-1)
    ref = np.asarray(ref_like) if ref_like is not None else None
    return arr.reshape(-1,1) if (ref is not None and ref.ndim==2) else arr

def RL_passive(ego_object, objects=None, *args, **kwargs):

    global _last_state, _step

    # current state
    dist, dth, e_r = _state_features(ego_object,**kwargs)
    state_now = _to_index(dist, dth)

    # reward
    r_now = -min(dist, D_MAX)   # the closer to the circle,the higher reward
    if getattr(ego_object, "collision", False):
        r_now -= 10.0


    # count（previous -> current state）
    if _last_state is not None:
        _trans[_last_state, state_now] += 1.0
        _vis[_last_state] += 1
        _Rsum[_last_state] += r_now  # sum of rewards attributed to _last_state

    _step += 1

    # Every "_ADP_EVERY" times give an evaluation
    if _step % _ADP_EVERY == 0:
        _adp_policy_evaluation()
        # print process
        print(f"[ADP] visited={int((_vis>0).sum())}/{_S}, "
              f"avgV={_V[_vis>0].mean():.3f}, "
              f"mean|dist|={getattr(RL_passive,'_stat',{}).get('md',0):.3f},steps={_step}")

    # constant policy
    v = _V0     # constant linear velocity
    w = - _KW * dth+ _KR * e_r  # angular velocity controlled by _KW and _KR * e_r

    # velocity constrained to vel_min/vel_max
    if hasattr(ego_object, "vel_min") and hasattr(ego_object, "vel_max"):
        vmin = np.asarray(ego_object.vel_min, np.float32).reshape(-1)
        vmax = np.asarray(ego_object.vel_max, np.float32).reshape(-1)
        v = float(np.clip(v, vmin[0], vmax[0]))
        w = float(np.clip(w, vmin[1], vmax[1]))

    _last_state = state_now
    _glob = getattr(RL_passive, "_stat", {"n": 0, "md": 0.0})
    _glob["n"] += 1
    _glob["md"] = 0.99 * _glob["md"] + 0.01 * dist    # exponential moving average
    setattr(RL_passive, "_stat", _glob)
    return _shape_like_ref([v, w], getattr(ego_object, "vel_min", None))




# ==== register behavior：（policy fixed，estimate V only） ====
@register_behavior("diff", "rl_passive")
def subsumption_nav(ego_object, objects=None, *args, **kwargs):
    v, w = RL_passive(ego_object, objects, *args, **kwargs)   #  RL_passive
    return np.array([v, w], dtype=np.float32)



