import argparse
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


HAND_BONES = [
    (0, 1), (1, 2), (1, 6), (1, 11), (1, 16), (1, 21),   # ладонь и основания
    (2, 3), (3, 4), (4, 5),                               # большой
    (6, 7), (7, 8), (8, 9), (9, 10),                      # указательный
    (11, 12), (12, 13), (13, 14), (14, 15),               # средний
    (16, 17), (17, 18), (18, 19), (19, 20),               # безымянный
    (21, 22), (22, 23), (23, 24), (24, 25),               # мизинец
]

NUMBER_RE = re.compile(r'-?\d+(?:,\d+)?')

def parse_pose(pose_str: str) -> np.ndarray:
    matches = NUMBER_RE.findall(pose_str)
    return np.array([float(m.replace(',', '.')) for m in matches], dtype=float)


def load_frames(path: Path):
    meta = None
    frames = []
    bad_lines = 0
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines += 1
                if bad_lines <= 3:
                    print(f'  WARN: skip line {lineno}: {e}')
                continue
            if 'Head' in obj or 'Hand' in obj:
                frames.append(obj)
            elif meta is None:
                meta = obj
    if bad_lines > 3:
        print(f'  WARN: skipped {bad_lines} malformed lines total')
    return meta, frames


def frame_time_s(frame: dict) -> float | None:
    ts = frame.get('timeStampNs')
    if ts is not None:
        return ts / 1e9
    pt = frame.get('predictTime')
    if pt is not None:
        return float(pt)
    return None

def head_position(frame: dict) -> np.ndarray | None:
    head = frame.get('Head')
    if not head or 'pose' not in head:
        return None
    pose = parse_pose(head['pose'])
    if pose.shape[0] < 3:
        return None
    return pose[:3]


def hand_joints(frame: dict, side: str) -> np.ndarray | None:
    hand = frame.get('Hand', {}).get(side, {})
    if not hand.get('isActive'):
        return None
    joints = hand.get('HandJointLocations', [])
    if not joints:
        return None
    pts = []
    for j in joints:
        p = parse_pose(j['p'])
        if p.shape[0] >= 3:
            pts.append(p[:3])
    if not pts:
        return None
    return np.asarray(pts, dtype=float)


def plot_hand_skeleton(points: np.ndarray, side: str, t_sec: float,
                       out_html: Path, out_png: Path | None):
    color = '#1f77b4' if side == 'leftHand' else '#d62728'
    label = 'Left hand' if side == 'leftHand' else 'Right hand'

    edge_x, edge_y, edge_z = [], [], []
    for a, b in HAND_BONES:
        if a < len(points) and b < len(points):
            edge_x += [points[a, 0], points[b, 0], None]
            edge_y += [points[a, 1], points[b, 1], None]
            edge_z += [points[a, 2], points[b, 2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#555', width=3),
        name='bones', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=5, color=color),
        text=[f'joint {i}' for i in range(len(points))],
        name=label,
    ))

    fig.update_layout(
        title=f'{label} skeleton — t = {t_sec:.2f} s',
        scene=dict(
            xaxis_title='X, m', yaxis_title='Y, m', zaxis_title='Z, m',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_html)
    print(f'  saved {out_html.name}')
    if out_png is not None:
        try:
            fig.write_image(out_png, width=900, height=700, scale=2)
            print(f'  saved {out_png.name}')
        except Exception as e:
            print(f'  WARN: PNG not saved ({e}). HTML still available.')


def plot_trajectories(heads, lefts, rights, out_html: Path, out_png: Path | None):
    fig = go.Figure()

    def add_traj(arr, name, color):
        if arr is None or len(arr) == 0:
            return
        arr = np.asarray(arr)
        if len(arr) == 1:
            fig.add_trace(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
                mode='markers',
                marker=dict(size=6, color=color),
                name=name,
            ))
            return
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode='lines',
            line=dict(color=color, width=4),
            name=name,
        ))
        fig.add_trace(go.Scatter3d(
            x=[arr[0, 0], arr[-1, 0]],
            y=[arr[0, 1], arr[-1, 1]],
            z=[arr[0, 2], arr[-1, 2]],
            mode='markers+text',
            marker=dict(size=6, color=color, symbol='diamond'),
            text=['start', 'end'],
            textposition='top center',
            name=f'{name} start/end',
            showlegend=False,
        ))

    add_traj(heads,  'Head',         '#2ca02c')
    add_traj(lefts,  'Left wrist',   '#1f77b4')
    add_traj(rights, 'Right wrist',  '#d62728')

    fig.update_layout(
        title='3D trajectories of head and wrists (whole recording)',
        scene=dict(
            xaxis_title='X, m', yaxis_title='Y, m', zaxis_title='Z, m',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_html)
    print(f'  saved {out_html.name}')
    if out_png is not None:
        try:
            fig.write_image(out_png, width=1000, height=800, scale=2)
            print(f'  saved {out_png.name}')
        except Exception as e:
            print(f'  WARN: PNG not saved ({e}). HTML still available.')


def path_length(arr) -> float:
    if arr is None or len(arr) < 2:
        return 0.0
    arr = np.asarray(arr)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def compute_stats(frames, heads, lefts, rights) -> dict:
    times = [t for f in frames if (t := frame_time_s(f)) is not None]
    duration = (max(times) - min(times)) if len(times) >= 2 else 0.0

    n_left_active = sum(
        1 for f in frames
        if f.get('Hand', {}).get('leftHand', {}).get('isActive')
    )
    n_right_active = sum(
        1 for f in frames
        if f.get('Hand', {}).get('rightHand', {}).get('isActive')
    )

    return {
        'frames': len(frames),
        'duration_s': round(duration, 3),
        'head_path_m': round(path_length(heads), 4),
        'left_wrist_path_m': round(path_length(lefts), 4),
        'right_wrist_path_m': round(path_length(rights), 4),
        'left_active_frames': n_left_active,
        'right_active_frames': n_right_active,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--outdir', default=Path('output'), type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_png = not args.no_png

    print(f'Loading {args.input} ...')
    meta, frames = load_frames(args.input)
    print(f'  meta present: {meta is not None}')
    print(f'  tracking frames: {len(frames)}')

    if not frames:
        raise SystemExit('No tracking frames found.')

    heads, lefts, rights = [], [], []
    first_left_frame = None
    first_right_frame = None
    for f in frames:
        hp = head_position(f)
        if hp is not None:
            heads.append(hp)

        lj = hand_joints(f, 'leftHand')
        if lj is not None:
            lefts.append(lj[0])
            if first_left_frame is None:
                first_left_frame = (f, lj)

        rj = hand_joints(f, 'rightHand')
        if rj is not None:
            rights.append(rj[0])
            if first_right_frame is None:
                first_right_frame = (f, rj)

    print(f'  head samples:  {len(heads)}')
    print(f'  left samples:  {len(lefts)}')
    print(f'  right samples: {len(rights)}')

    stats = compute_stats(frames, heads, lefts, rights)
    stats_path = args.outdir / 'stats.json'
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print('Statistics:')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    if first_left_frame is not None:
        f, pts = first_left_frame
        t = frame_time_s(f) or 0.0
        plot_hand_skeleton(
            pts, 'leftHand', t,
            args.outdir / 'hand_skeleton_left.html',
            args.outdir / 'hand_skeleton_left.png' if save_png else None,
        )
    elif first_right_frame is not None:
        f, pts = first_right_frame
        t = frame_time_s(f) or 0.0
        plot_hand_skeleton(
            pts, 'rightHand', t,
            args.outdir / 'hand_skeleton_right.html',
            args.outdir / 'hand_skeleton_right.png' if save_png else None,
        )
    else:
        print('  no active hand frames — skipping skeleton plot')

    if heads or lefts or rights:
        plot_trajectories(
            heads, lefts, rights,
            args.outdir / 'trajectories.html',
            args.outdir / 'trajectories.png' if save_png else None,
        )
    else:
        print('  no trajectory data — skipping trajectory plot')

    print(f'\nDone. Outputs in {args.outdir.resolve()}')


if __name__ == '__main__':
    main()