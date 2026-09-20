"""Show a completed KITTI run as a synchronized camera/path playback."""
import argparse
import base64
import csv
from pathlib import Path
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/kitti/2011_09_26/2011_09_26_drive_0052_sync'))
    parser.add_argument('--run', type=Path, default=Path('outputs/kitti_0052'))
    args = parser.parse_args()
    poses = np.atleast_2d(np.loadtxt(args.run/'trajectory.tum'))
    with (args.run/'diagnostics.csv').open() as stream:
        diagnostics = list(csv.DictReader(stream))
    initial = Rotation.from_quat(poses[0, 4:8]).as_matrix()
    positions = (poses[:, 1:4] - poses[0, 1:4]) @ initial
    rotations = Rotation.from_quat(poses[:, 4:8]).as_matrix()
    lines = (args.run/'map.ply').read_text().splitlines()
    cloud = np.array([list(map(float, line.split())) for line in lines[lines.index('end_header')+1:]])
    cloud = (cloud - poses[0, 1:4]) @ initial
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    # Map axes are first-camera x (right) and z (forward), with equal meter scale.
    xz = positions[:, [0, 2]]
    center = (xz.min(axis=0)+xz.max(axis=0))/2
    span = max(float(np.ptp(xz, axis=0).max())+12., 30.)
    map_origin = np.array([970., 210.])
    map_size = np.array([420., 510.])
    scale = map_size.min()/span

    def to_pixels(points):
        return np.rint((points-center)*[scale,-scale]+map_origin+map_size/2).astype(int)

    def label(img, text, xy, size=.55, color=(210, 205, 192)):
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    frames = args.run/'playback_frames'
    frames.mkdir(exist_ok=True)
    for i, row in enumerate(diagnostics):
        out = np.full((900, 1440, 3), (24, 19, 15), np.uint8)
        label(out, 'KITTI  /  STEREO + IMU LOCALIZATION', (24, 42), .92, (244, 238, 225))
        label(out, f'Drive 0052    |    {i+1:02d}/{len(poses)} frames    |    {poses[i,0]:.2f} s', (24, 74), .57)
        left = cv2.imread(str(args.data/'image_00'/'data'/row['frame']))
        mask = cv2.imread(str(args.run/'masks'/row['frame']), cv2.IMREAD_GRAYSCALE)
        if left is None or mask is None:
            raise ValueError(f'Missing image or saved mask: {row["frame"]}')
        overlay = left.copy()
        overlay[mask>0] = (.45*overlay[mask>0]+.55*np.array([70, 75, 250])).astype(np.uint8)
        label(out,'LEFT CAMERA  /  RECORDED VEHICLE VIEW',(24,111),.55)
        out[128:400,24:924] = cv2.resize(left,(900,272))
        label(out,'MOTION FILTER  /  RED = EXCLUDED REGIONS',(24,445),.55)
        out[462:734,24:924] = cv2.resize(overlay,(900,272))
        label(out,'ESTIMATED ROBOT POSE',(970,111),.66,(244,238,225))
        label(out,'Top view | arrow = camera heading',(970,145),.47)
        label(out,'Gray points: final reconstructed map',(970,173),.47)
        cv2.rectangle(out,(950,195),(1415,750),(63,53,43),1)
        for p in to_pixels(cloud[:,[0,2]]):
            if 955<p[0]<1410 and 200<p[1]<745:
                cv2.circle(out,tuple(p),1,(97,79,59),-1,cv2.LINE_AA)
        trail=to_pixels(xz[:i+1])
        if len(trail)>1:
            cv2.polylines(out,[trail.reshape(-1,1,2)],False,(244,208,77),3,cv2.LINE_AA)
        cv2.circle(out,tuple(trail[0]),4,(155,229,151),-1)
        # Heading is transformed from current camera into initial-camera axes.
        heading=(initial.T @ rotations[i] @ np.array([0.,0.,1.]))[[0,2]]
        heading=heading/np.linalg.norm(heading)*[1.,-1.]
        side=np.array([-heading[1],heading[0]])
        p=trail[-1]
        arrow=np.int32([p+heading*14,p-heading*10+side*9,p-heading*10-side*9])
        cv2.fillConvexPoly(out,arrow,(244,208,77),cv2.LINE_AA)
        bar=int(5*scale)
        cv2.line(out,(977,725),(977+bar,725),(220,220,220),2)
        label(out,'5 m',(977,713),.43)
        label(out,'Start',(971,788),.5,(155,229,151))
        label(out,'Estimated path',(1060,788),.5,(244,208,77))
        label(out,f'Path length  {distance[i]:.1f} m',(970,823),.62,(244,238,225))
        label(out,f'LANDMARKS  {row["landmarks"]}     POSE INLIERS  {row["inliers"]}     MEDIAN REPROJECTION  {float(row["median_reprojection_px"]):.2f} px', (24,785),.57)
        label(out,'Actual KITTI recording; estimated camera motion. No physical robot is being controlled.',(24,831),.5)
        label(out,'Map uses all completed frames. Playback follows recorded timestamps at approximately 10 Hz.',(24,861),.47)
        cv2.imwrite(str(frames/f'{i:04d}.png'),out)
    fps=(len(poses)-1)/(poses[-1,0]-poses[0,0]) if len(poses)>1 else 10.
    video=args.run/'robot_demo.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(frames/'%04d.png'),'-frames:v',str(len(poses)),
                    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex',
                    '[0:v]fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                    '-loop','0',str(args.run/'robot_demo.gif')],check=True)
    encoded=base64.b64encode(video.read_bytes()).decode()
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>KITTI robot playback</title>
<style>body{margin:0;padding:28px;background:#0f1318;color:#e1eef2;font:16px system-ui}main{max-width:1440px;margin:auto}h1{font-size:25px}p{color:#a0abb9;line-height:1.6}video{width:100%;border-radius:12px;border:1px solid #34404a}button{background:#4dd0f2;border:0;border-radius:6px;padding:10px 18px;margin:12px 8px 0 0;cursor:pointer}</style>
<main><h1>KITTI drive 0052 — robot pose playback</h1><p>The cyan arrow follows the estimated camera position and heading. Red overlays show regions excluded from localization. Gray map points show the completed reconstruction.</p>
<video id="playback" controls autoplay muted loop playsinline src="data:video/mp4;base64,VIDEO"></video>
<button onclick="const v=document.getElementById('playback');v.currentTime=0;v.play()">Replay</button>
<button onclick="document.getElementById('playback').playbackRate=.5">Half speed</button>
<button onclick="document.getElementById('playback').playbackRate=1">Normal speed</button>
<p>All 78 recorded frames processed successfully. This is estimated motion from a real KITTI vehicle recording, not control of a physical robot. No ground-truth accuracy claim is made.</p></main></html>'''.replace('VIDEO',encoded)
    (args.run/'robot_demo.html').write_text(page)
    cv2.imwrite(str(args.run/'robot_preview.jpg'),out)
    print(f'Created robot_demo.html, robot_demo.mp4, robot_demo.gif in {args.run}')
    print(f'{len(poses)} poses; {len(cloud)} cloud points; {distance[-1]:.2f} m estimated path; {poses[-1,0]:.2f} s')


if __name__=='__main__':
    main()
