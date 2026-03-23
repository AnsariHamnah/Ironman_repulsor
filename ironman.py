import cv2
import numpy as np
import mediapipe as mp
import time
import math
import random

# ---------------- MediaPipe Setup ----------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ---------------- Globals ----------------
charge = 0
angle = 0
particles = []

# smoothing
prev_x, prev_y = 300, 300

# ---------------- Particle ----------------
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-10, 10)
        self.vy = random.uniform(-10, 10)
        self.life = 25

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

# ---------------- Plasma Ball ----------------
def draw_repulsor(frame, x, y, charge):

    for i in range(10):
        radius = 50 + i*6 + random.randint(-4,4)
        color = (255, 180 + i*8, 50)

        overlay = frame.copy()
        cv2.circle(overlay, (x,y), radius, color, -1)

        alpha = 0.06
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # core
    cv2.circle(frame, (x,y), 18, (255,255,255), -1)

# ---------------- Beam ----------------
def draw_beam(frame, x, y):

    for i in range(6):
        thickness = 14 - i*2
        color = (255, 200 + i*10, 50)

        cv2.line(frame, (x,y), (x,y-350), color, thickness)

    cv2.line(frame, (x,y), (x,y-350), (255,255,255), 3)

# ---------------- HUD Rings ----------------
def draw_hud(frame, x, y, angle):

    for i in range(4):
        radius = 70 + i*25

        for a in range(0,360,20):
            rad = math.radians(a + angle*(i+1))
            px = int(x + radius * math.cos(rad))
            py = int(y + radius * math.sin(rad))

            cv2.circle(frame,(px,py),2,(255,200,50),-1)

# ---------------- Arc Reactor ----------------


# ---------------- Gesture Detection ----------------
def is_fist(hand):
    # compare fingertip vs base
    return hand[8].y > hand[6].y

# ---------------- Main Loop ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
        timestamp
    )

    h, w, _ = frame.shape
    cx, cy = prev_x, prev_y

    firing = False

    # -------- Hand Tracking --------
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        raw_x = int(hand[9].x * w)
        raw_y = int(hand[9].y * h)

        # smoothing
        cx = int(prev_x * 0.7 + raw_x * 0.3)
        cy = int(prev_y * 0.7 + raw_y * 0.3)

        prev_x, prev_y = cx, cy

        # gesture
        if is_fist(hand):
            firing = True
        else:
            charge = min(150, charge + 2)

    else:
        charge = max(0, charge - 5)

    # -------- Fire Logic --------
    if firing and charge > 60:

        draw_beam(frame, cx, cy)

        for _ in range(60):
            particles.append(Particle(cx, cy))

        charge = 0

    # -------- Draw Effects --------
    draw_repulsor(frame, cx, cy, charge)
    draw_hud(frame, cx, cy, angle)
    

    angle += 2

    # -------- Particles --------
    for p in particles:
        p.update()
        if p.life > 0:
            cv2.circle(frame,(int(p.x),int(p.y)),3,(255,255,255),-1)

    particles = [p for p in particles if p.life > 0]

    # -------- Jarvis UI --------
    cv2.putText(frame,"JARVIS ONLINE",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,200,50),2)

    cv2.putText(frame,f"CHARGE: {charge}",(20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,200,50),1)

    # -------- Cinematic Glow --------
    blur = cv2.GaussianBlur(frame,(0,0),11)
    frame = cv2.addWeighted(frame,1,blur,0.6,0)

    cv2.imshow("IRON MAN FINAL SYSTEM", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()