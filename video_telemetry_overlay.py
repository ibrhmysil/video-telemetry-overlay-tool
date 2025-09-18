#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 by Ibrahim Yesil . All rights reserved. This file is part
# of the Merih Space CanSat project, released under the MIT License. Please
# see the LICENSE file included as part of this package.

"""
video_telemetry_overlay.py

Usage scenarios:
-------------------
# 1) Synchronized by timestamp (CSV contains ‘Send Time’)
python video_telemetry_overlay.py --video flight.avi --csv telemetry.csv \
  --csv-time-col “Send Time” --csv-time-format “%Y-%m-%d %H:%M:%S.%f” \
  --video-start “2025-09-12 14:03:05.200” --out out.mp4

# 2) Synchronize by packet number (align the start of the video with packet 12345 in the CSV)
python video_telemetry_overlay.py --video flight.avi --csv telemetry.csv \
  --packet-start 12345 --out out.mp4

# 3) Only with manual offset (videonun karesi t_video, CSV zamanı t_csv + offset)
python video_telemetry_overlay.py --video flight.avi --csv telemetri.csv \
  --offset 2.4 --out out.mp4

Keys in preview mode:
  j / l : ofseti -0.1s / +0.1s
  J / L : ofseti -1.0s / +1.0s
  s     : dosyaya render al (baştan sona)
  q     : çık
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import re
from dateutil import parser as dtparser
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
import sys
import math
import os

# ---- Customization section by column names ----
DEFAULT_COLS = {
    "packet": ["Paket Numarasi", "Paket", "PaketNo", "Packet", "Paket_Numarasi"],
    "status": ["Uydu Statusu", "Uydu Durumu", "Status"],
    "error":  ["Hata Kodu", "Hata", "Error"],
    "time":   ["Gonderme Saati", "Zaman", "Time", "Timestamp"],
    "p1":     ["P1", "Basinc1", "Basınç1", "Pressure1"],
    "p2":     ["P2", "Basinc2", "Basınç2", "Pressure2"],
    "alt1":   ["Yukseklik 1", "Yükseklik 1"],
    "alt2":   ["Yukseklik 2", "Yükseklik 2"],
    "altd":   ["Irtifa Farki", "İrtifa Farkı"],
    "speed":  ["Inis Hizi", "İnis Hizi", "İniş Hızı", "VerticalSpeed", "Speed"],
    "temp":   ["Temp", "Sicaklik", "Sıcaklık", "Temperature"],
    "voltage":["Pil Gerilimi", "Voltage", "Vbat", "Battery"],
    "lat":    ["Lat", "GPS Latitude", "Latitude"],
    "lon":    ["Lon", "GPS Longitude", "Longitude"],
    "alt":    ["Alt", "GPS Alt", "GPS Altitude", "Altitude"],
    "pitch":  ["Pitch"],
    "roll":   ["Roll"],
    "yaw":    ["Yaw"],
    "rhrh":   ["RHRH", "humidity", "Nem"],
    "iot1":   ["IoT S1 Data", "IoT1"],
    "iot2":   ["IoT S2 Data", "IoT2"],
    "team":   ["Takim Numarasi", "Team"],
}

# Keys to be entered into the overlay numerically
NUMERIC_KEYS = ["p1", "p2", "alt1", "alt2", "altd", "speed", "temp", "voltage",
                "pitch", "roll", "yaw", "lat", "lon", "alt", "iot1", "iot2"]

# ---------- Header drawing (with Pillow, Turkish language support) ----------
def draw_center_title(frame, text, y_ratio=0.09, font_size=64,
                      font_path="/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                      fill=(255,255,255), stroke_width=6, stroke_fill=(0,0,0)):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (w - text_w) // 2
    y = int(h * y_ratio) - text_h // 2
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

# ---------- CSV helpers ----------
def guess_sep_and_decimal(sample_path, n_lines=20):
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        head = "".join([next(f, "") for _ in range(n_lines)])
    if head.count("\t") > head.count(",") and head.count("\t") > head.count(";"):
        sep = "\t"
    elif head.count(";") > head.count(","):
        sep = ";"
    else:
        sep = ","
    comma_decimal = re.search(r"\d+,\d+", head) is not None
    dot_decimal   = re.search(r"\d+\.\d+", head) is not None
    if comma_decimal and not dot_decimal:
        decimal = ","
    else:
        decimal = "."
    return sep, decimal

def read_csv_flex(path):
    sep, decimal = guess_sep_and_decimal(path)
    try:
        df = pd.read_csv(path, sep=sep, decimal=decimal, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_numeric_series(s):
    if s.dtype == object:
        s2 = s.astype(str).str.strip()
        s2 = s2.replace({"Null": np.nan, "NULL": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
        s2 = s2.str.replace(",", ".", regex=False)
        return pd.to_numeric(s2, errors="coerce")
    else:
        return pd.to_numeric(s, errors="coerce")

def build_telemetry_timeseries(df, args, colmap):
    n = len(df)
    t_csv = np.zeros(n, dtype=np.float64)

    if args.csv_time_col or (colmap["time"] is not None and args.video_start):
        time_col = args.csv_time_col or colmap["time"]
        def parse_dt(x):
            x = str(x).strip()
            if args.csv_time_format:
                return datetime.strptime(x, args.csv_time_format)
            else:
                return dtparser.parse(x)
        times = df[time_col].apply(parse_dt)
        t0 = dtparser.parse(args.video_start) if args.video_start else times.iloc[0]
        t_csv = (times - t0).dt.total_seconds().to_numpy()
    elif args.packet_start or colmap["packet"]:
        pcol = colmap["packet"]
        if pcol is None:
            t_csv = np.arange(n) * (args.sample_dt or 0.1)
        else:
            packets = normalize_numeric_series(df[pcol]).to_numpy()
            ref_packet = args.packet_start if args.packet_start is not None else packets[0]
            packet_dt = args.packet_dt or 0.1
            t_csv = (packets - ref_packet) * packet_dt
    else:
        dt = args.sample_dt or 0.1
        t_csv = np.arange(n) * dt

    return t_csv

def interpolate_row_at_time(t_csv, df_num, t):
    if len(t_csv) == 0:
        return {k: np.nan for k in df_num.columns}
    if t <= t_csv[0]:
        return df_num.iloc[0].to_dict()
    if t >= t_csv[-1]:
        return df_num.iloc[-1].to_dict()
    idx = np.searchsorted(t_csv, t)
    i0 = max(0, idx - 1)
    i1 = min(len(t_csv) - 1, idx)
    t0, t1 = t_csv[i0], t_csv[i1]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    v0 = df_num.iloc[i0]
    v1 = df_num.iloc[i1]
    out = {}
    for c in df_num.columns:
        a, b = v0[c], v1[c]
        if np.isnan(a) and np.isnan(b):
            out[c] = np.nan
        elif np.isnan(a):
            out[c] = b
        elif np.isnan(b):
            out[c] = a
        else:
            out[c] = (1 - w) * a + w * b
    return out

# ---------- Take string/discrete fields from the nearest row ----------
def discrete_text_at_time(t_csv, df_raw, colmap, t):
    if len(t_csv) == 0:
        return {}
    idx = int(np.clip(np.searchsorted(t_csv, t), 0, len(t_csv)-1))
    if idx > 0 and idx < len(t_csv):
        if abs(t_csv[idx-1] - t) <= abs(t_csv[idx] - t):
            idx = idx - 1

    out = {}
    wanted = {
        "time":   ["time", "Gonderme Saati", "Zaman", "Time", "Timestamp"],
        "packet": ["packet", "Paket", "Paket Numarasi", "PaketNo", "Packet", "Paket_Numarasi"],
        "status": ["status", "Uydu Statusu", "Uydu Durumu", "Status"],
        "error":  ["error", "Hata Kodu", "Hata", "Error"],
        "team":   ["team", "Takim Numarasi", "Team"],
        "rhrh":   ["rhrh", "RHRH", "humidity", "Nem"],
    }

    def resolve_col(cands):
        for k in cands:
            if k in colmap and colmap[k] and colmap[k] in df_raw.columns:
                return colmap[k]
        for k in cands:
            if k in df_raw.columns:
                return k
        return None

    for std_key, cands in wanted.items():
        colname = resolve_col(cands)
        if colname is not None:
            out[std_key] = df_raw.iloc[idx][colname]
    return out

# ---------- Attitude Indicator (horizon + roll) ----------
def draw_attitude_indicator(frame, cx, cy, radius, pitch_deg, roll_deg):
    # Outer ring
    cv2.circle(frame, (cx, cy), radius, (255,255,255), 2, cv2.LINE_AA)

    # Horizon: roll turn + pitch offset
    angle = np.deg2rad(roll_deg if not math.isnan(roll_deg) else 0.0)
    pitch = 0.0 if math.isnan(pitch_deg) else np.clip(pitch_deg, -30, 30)
    pitch_offset = int((pitch / 30.0) * (radius * 0.6))

    dx = int(radius * np.cos(angle))
    dy = int(radius * np.sin(angle))
    cv2.line(frame, (cx - dx, cy + dy + pitch_offset),
             (cx + dx, cy - dy + pitch_offset), (255,255,255), 3, cv2.LINE_AA)
    cv2.line(frame, (cx-12, cy + pitch_offset), (cx+12, cy + pitch_offset), (0,0,0), 5, cv2.LINE_AA)
    cv2.line(frame, (cx-12, cy + pitch_offset), (cx+12, cy + pitch_offset), (0,255,0), 2, cv2.LINE_AA)

    # Roll scale
    for deg in range(-90, 91, 10):
        a = np.deg2rad(deg)
        sx = int(cx - (radius+6) * np.sin(a))
        sy = int(cy - (radius+6) * np.cos(a))
        ex = int(cx - (radius-6) * np.sin(a))
        ey = int(cy - (radius-6) * np.cos(a))
        thickness = 2 if deg % 30 == 0 else 1
        cv2.line(frame, (sx, sy), (ex, ey), (255,255,255), thickness, cv2.LINE_AA)

# ---------- Yaw Heading Tape ----------
def draw_yaw_tape_center(frame, yaw_deg, y_bottom_offset, width_ratio, tape_h):
    h, w = frame.shape[:2]
    yaw = 0.0 if (yaw_deg is None or (isinstance(yaw_deg, float) and math.isnan(yaw_deg))) else (yaw_deg % 360.0)

    # width_ratio slaced, 120px to w-40
    tape_w = int(np.clip(w * width_ratio, 120, w - 40))
    tx0 = (w - tape_w) // 2
    ty0 = h - y_bottom_offset - tape_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (tx0, ty0), (tx0 + tape_w, ty0 + tape_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cx = w // 2
    cv2.line(frame, (cx, ty0), (cx, ty0 + tape_h), (255,255,255), 2, cv2.LINE_AA)

    def yaw_label_exact(d):
        d = int((d + 360) % 360)
        return {0:"N",90:"E",180:"S",270:"W"}.get(d, str(d))

    half_span = 90
    for d in range(-half_span, half_span+1, 10):
        val = (yaw + d) % 360
        x = int(cx + (d / half_span) * (tape_w//2 - 12))
        if d % 30 == 0:
            cv2.line(frame, (x, ty0), (x, ty0 + tape_h), (210,210,210), 1, cv2.LINE_AA)
            key = round(val/30)*30
            label = yaw_label_exact(key)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, label, (x - lw//2, ty0 + tape_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        else:
            cv2.line(frame, (x, ty0 + tape_h//2), (x, ty0 + tape_h), (160,160,160), 1, cv2.LINE_AA)

# ---------- GPS Mini Map (black-white gird) ----------
def _latlon_to_xy_m(lat0, lon0, lat, lon):
    # simple metric convert (equirectangular)
    k = 111320.0
    x = (lon - lon0) * k * math.cos(math.radians(lat0))
    y = (lat - lat0) * k * 1.0
    return x, y

def draw_gps_minimap(frame, t_csv_all, lat_all, lon_all, t_query,
                     box_w=220, box_h=220, margin=16, right_bar_x=None, yaw_tape_bottom=0):
    h, w = frame.shape[:2]
    if t_csv_all is None or lat_all is None or lon_all is None:
        return
    mask = (t_csv_all <= t_query)
    if not np.any(mask):
        return
    t_hist = t_csv_all[mask]
    lat_hist = lat_all[mask]
    lon_hist = lon_all[mask]

    T = 60.0
    mask2 = (t_hist >= (t_query - T))
    if not np.any(mask2):
        mask2 = slice(None)
    lat_hist = lat_hist[mask2]
    lon_hist = lon_hist[mask2]

    good = np.isfinite(lat_hist) & np.isfinite(lon_hist)
    if not np.any(good):
        return
    lat_hist = lat_hist[good]
    lon_hist = lon_hist[good]

    lat0 = np.median(lat_hist)
    lon0 = np.median(lon_hist)
    xs, ys = [], []
    for la, lo in zip(lat_hist, lon_hist):
        x, y = _latlon_to_xy_m(lat0, lon0, la, lo)
        xs.append(x); ys.append(y)
    xs = np.array(xs); ys = np.array(ys)

    if xs.size < 2:
        x_min, x_max = -10.0, 10.0
        y_min, y_max = -10.0, 10.0
    else:
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        pad_x = 0.2 * max(1.0, (x_max - x_min))
        pad_y = 0.2 * max(1.0, (y_max - y_min))
        x_min -= pad_x; x_max += pad_x
        y_min -= pad_y; y_max += pad_y

    if right_bar_x is None:
        right_bar_x = w - 130
    x1 = right_bar_x - 24
    x0 = x1 - box_w
    if yaw_tape_bottom <= 0:
        y1 = h - margin
    else:
        y1 = yaw_tape_bottom - 10
    y0 = y1 - box_h
    x0 = max(margin, x0)
    y0 = max(margin, y0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0-6, y0-6), (x1+6, y1+6), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255,255,255), 2, cv2.LINE_AA)

    # Grid 4x4
    for gx in [0.25, 0.5, 0.75]:
        xx = int(x0 + gx * (box_w))
        cv2.line(frame, (xx, y0), (xx, y1), (180,180,180), 1, cv2.LINE_AA)
    for gy in [0.25, 0.5, 0.75]:
        yy = int(y0 + gy * (box_h))
        cv2.line(frame, (x0, yy), (x1, yy), (180,180,180), 1, cv2.LINE_AA)

    def to_px(xm, ym):
        u = 0 if (x_max - x_min) == 0 else (xm - x_min) / (x_max - x_min)
        v = 0 if (y_max - y_min) == 0 else (ym - y_min) / (y_max - y_min)
        px = int(x0 + u * (box_w))
        py = int(y1 - v * (box_h))
        return px, py

    pts = [to_px(xm, ym) for xm, ym in zip(xs, ys)]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (255,255,255), 2, cv2.LINE_AA)

    cx, cy = pts[-1]
    cv2.circle(frame, (cx, cy), 4, (255,255,255), -1, cv2.LINE_AA)

    # Scale bar
    span_m = max(x_max - x_min, y_max - y_min)
    if span_m <= 50:
        scalem = 50
    elif span_m <= 100:
        scalem = 20
    elif span_m <= 200:
        scalem = 50
    else:
        scalem = 100
    px_a, _ = to_px(x_min, y_min)
    px_b, _ = to_px(x_min + scalem, y_min)
    #cv2.line(frame, (px_a + 10, y1 - 10), (px_b + 10, y1 - 10), (255,255,255), 2, cv2.LINE_AA)
    label = f"{scalem} m"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, label, (px_a + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

# ---------- Panel Drawing + Altitude Bar + Colorful Top-Right + Yaw + MiniMap + Logo ----------
def draw_panel(frame, hud, t_video, fps, offset, alt_minmax=None,
               t_csv_all=None, lat_all=None, lon_all=None):
    import math
    import numpy as np
    import cv2

    h, w = frame.shape[:2]

    # ---------- Helpers ----------
    NULL_STRINGS = {"null", "NULL", "Null", "Nan", "nan", "None", ""}

    def is_nan(x):
        try:
            return x is None or (isinstance(x, float) and math.isnan(x))
        except:
            return False

    def is_nullish(x):
        if x is None:
            return True
        if isinstance(x, (float, int, np.floating, np.integer)):
            return is_nan(float(x))
        try:
            s = str(x).strip()
        except Exception:
            return False
        return s in NULL_STRINGS

    def to_float_or_nan(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    def fmt(x, fmt_str="{:.2f}"):
        if is_nullish(x):
            return "--"
        if isinstance(x, (float, int, np.floating, np.integer)):
            try:
                return fmt_str.format(float(x))
            except Exception:
                return str(x)
        try:
            xf = float(str(x).replace(",", "."))
            if math.isnan(xf):
                return "--"
            return fmt_str.format(xf)
        except Exception:
            return str(x)

    def get_any(d, keys, default=None):
        for k in keys:
            if k in d and not is_nullish(d[k]):
                return d[k]
        return default

    def draw_boxed_texts(anchor, lines, align="left", box_alpha=0.45, font_scale=0.6, thickness=2, pad=10):
        lh = 26
        text_w = 0
        for ln in lines:
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_w = max(text_w, tw)
        text_h = lh * len(lines)

        if align == "left":
            x0, y0 = anchor
        elif align == "right":
            x0, y0 = anchor[0] - (text_w + 2*pad), anchor[1]
        else:
            x0, y0 = anchor[0] - (text_w//2 + pad), anchor[1]

        x1, y1 = x0 + text_w + 2*pad, y0 + text_h + 2*pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)

        y = y0 + pad + 20
        for ln in lines:
            cv2.putText(frame, ln, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            y += lh

        return (x0, y0, x1, y1)

    def draw_boxed_texts_colored(anchor, items, align="left", box_alpha=0.45, font_scale=0.6, thickness=2, pad=10):
        """
        items: [(text, (B,G,R)), ...]
        """
        lh = 26
        text_w = 0
        for ln, _ in items:
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_w = max(text_w, tw)
        text_h = lh * len(items)

        if align == "left":
            x0, y0 = anchor
        elif align == "right":
            x0, y0 = anchor[0] - (text_w + 2*pad), anchor[1]
        else:
            x0, y0 = anchor[0] - (text_w//2 + pad), anchor[1]

        x1, y1 = x0 + text_w + 2*pad, y0 + text_h + 2*pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)

        y = y0 + pad + 20
        for ln, color in items:
            cv2.putText(frame, ln, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        color, thickness, cv2.LINE_AA)
            y += lh

        return (x0, y0, x1, y1)

    # ---------- UPPER CENTER HEADING ----------
    title = "KOKLERDEN GOKLERE"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    tx = (w - tw) // 2
    ty = int(0.09 * h)
    cv2.putText(frame, title, (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    subtitle = "TEKNOFEST 2025 Model Uydu Yarismasi"
    (tw2, th2), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    tx2 = (w - tw2) // 2
    cv2.putText(frame, subtitle, (tx2, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (tx2, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    subtitle = "Merih Space"
    (tw3, th3), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    tx3 = (w - tw3) // 2
    cv2.putText(frame, subtitle, (tx3, ty + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (tx3, ty + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # ---------- COLLECT DATA ----------
    time_txt   = get_any(hud, ["time", "timestamp", "Gonderme Saati", "Zaman"])
    if is_nullish(time_txt):
        time_txt = f"{t_video:0.2f}s"
    packet_val = get_any(hud, ["packet", "Paket", "Paket Numarasi", "PaketNo"])
    status_val = get_any(hud, ["status", "Uydu Statusu", "Status"])
    error_val  = get_any(hud, ["error", "Hata Kodu", "Hata", "Error"])
    team_val   = get_any(hud, ["team", "Takim Numarasi", "Tno", "Team"])

    left_top_lines = [
        f"Time: {fmt(time_txt, '{}')}   UTC+3",
        f"Packet No: {fmt(packet_val, '{:.0f}')}",
        f"Status: {fmt(status_val, '{:.0f}')}",
        f"Error Code: {fmt(error_val, '{:.0f}')}",
        f"Team ID: {fmt(team_val, '{:.0f}')}",
    ]

    voltage_val = get_any(hud, ["voltage", "Pil Gerilimi", "Vbat", "battery"])
    temp_val    = get_any(hud, ["temp", "Temperature", "Sicaklik", "Sıcaklık"])
    rhrh_val    = get_any(hud, ["rhrh", "RHRH"])

    # --- Temperature bar color ---
    temp_f = to_float_or_nan(temp_val)
    if math.isnan(temp_f):
        temp_color = (255,255,255)
    elif temp_f < 25.0:
        temp_color = (255,100,0)
    elif temp_f <= 35.0:
        temp_color = (0,255,0)
    else:
        temp_color = (0,0,255)

    # --- filter command color---
    filter_present = not is_nullish(rhrh_val)
    rhrh_color = (0,255,0) if filter_present else (255,255,255)

    # Top-right box: colored text
    right_top_items = [
        (f"Voltage: {fmt(voltage_val, '{:.2f}')} V", (255,255,255)),
        (f"Temperature: {fmt(temp_val, '{:.1f}')} C", temp_color),
        (f"Filter Command: {fmt(rhrh_val, '{:.1f}')}", rhrh_color),
    ]

    # ---------- Draw blocks ----------
    margin = 16
    draw_boxed_texts(anchor=(margin, margin), lines=left_top_lines, align="left",
                     box_alpha=0.35, font_scale=0.6, thickness=2, pad=10)
    rt_box = draw_boxed_texts_colored(anchor=(w - margin, margin), items=right_top_items, align="right",
                                      box_alpha=0.45, font_scale=0.6, thickness=2, pad=10)

    pitch_val = get_any(hud, ["pitch"])
    roll_val  = get_any(hud, ["roll"])
    yaw_val   = get_any(hud, ["yaw"])
    left_bottom_lines = [
        f"Pitch: {fmt(pitch_val, '{:.1f}')}",
        f"Roll: {fmt(roll_val,  '{:.1f}')}",
        f"Yaw: {fmt(yaw_val,   '{:.1f}')}",
    ]

    lat_val   = get_any(hud, ["lat", "Latitude"])
    lon_val   = get_any(hud, ["lon", "Longitude"])
    alt_val   = get_any(hud, ["alt", "GPS Alt", "Altitude"])
    speed_val = get_any(hud, ["speed", "vdown", "Inis Hizi", "İniş Hızı", "VerticalSpeed"])
    alt1_val  = get_any(hud, ["alt1", "Yukseklik 1", "Yükseklik 1"])
    alt2_val  = get_any(hud, ["alt2", "Yukseklik 2", "Yükseklik 2"])
    altd_val  = get_any(hud, ["altd", "Irtifa Farki", "İrtifa Farkı"])

    right_bottom_lines = [
        f"Vertical Speed: {fmt(speed_val,'{:.1f}')} m/s",
        f"Payload Alt.: {fmt(alt1_val, '{:.1f}')} m",
        f"Container Alt.: {fmt(alt2_val, '{:.1f}')} m",
        f"Alt. Difference: {fmt(altd_val,'{:.1f}')} m",
        f"GPS Latitude: {fmt(lat_val,  '{:.6f}')}",
        f"GPS Longitude: {fmt(lon_val,  '{:.6f}')}",
        f"GPS Altitude: {fmt(alt_val,  '{:.1f}')} m",
    ]

    lb_lh = 26; lb_pad = 10; lb_h = lb_pad*2 + lb_lh*len(left_bottom_lines)
    lb_box = draw_boxed_texts(anchor=(margin, h - margin - lb_h), lines=left_bottom_lines, align="left",
                              box_alpha=0.45, font_scale=0.6, thickness=2, pad=lb_pad)

    rb_lh = 26; rb_pad = 10; rb_h = rb_pad*2 + rb_lh*len(right_bottom_lines)
    draw_boxed_texts(anchor=(w - margin, h - margin - rb_h), lines=right_bottom_lines, align="right",
                     box_alpha=0.45, font_scale=0.6, thickness=2, pad=rb_pad)

    # ---------- Battery Bar (just below the top right box) ----------
    (rt_x0, rt_y0, rt_x1, rt_y1) = rt_box
    volt = to_float_or_nan(voltage_val)
    bx1 = w - margin - 75
    bar_w = 80
    bar_h = 20
    bx0 = bx1 - bar_w
    by0 = rt_y1 + 30
    by1 = by0 + bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx0-8, by0-8), (bx1, by1+8), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (255,255,255), 2, cv2.LINE_AA)

    V_MIN = 6.0
    V_MAX = 8.4
    if not math.isnan(volt):
        pct = np.clip((volt - V_MIN) / (V_MAX - V_MIN), 0.0, 1.0)
        fill_w = int((bar_w - 4) * pct)
        if pct < 0.20:
            fill_color = (0,0,255)
        elif pct < 0.50:
            fill_color = (0,255,255)
        else:
            fill_color = (0,255,0)
        cv2.rectangle(frame, (bx0+2, by0+2), (bx0+2+fill_w, by1-2), fill_color, -1, cv2.LINE_AA)
        txt = f"{volt:.2f} V  ({int(round(pct*100)):d}%)"
    else:
        txt = "N/A"
    (twb, thb), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, txt, (bx0 + (bar_w - twb)//2, by0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,255,255), 1, cv2.LINE_AA)

    # ---------- ALTITUDE BAR (center right) ----------
    cur_alt = to_float_or_nan(alt_val)
    cur_alt1 = to_float_or_nan(alt1_val) if not is_nullish(alt1_val) else float("nan")
    cur_alt2 = to_float_or_nan(alt2_val) if not is_nullish(alt2_val) else float("nan")

    if alt_minmax is not None:
        a_min = 0
        a_max = 700
    else:
        vals = [v for v in [cur_alt, cur_alt1, cur_alt2] if not math.isnan(v)]
        if len(vals) == 0:
            a_min, a_max = 0.0, 700.0
        else:
            a_min, a_max = min(vals), max(vals)
            if abs(a_max - a_min) < 1e-6:
                a_min -= 1.0; a_max += 1.0
            pad_v = 0.05 * (a_max - a_min)
            a_min -= pad_v; a_max += pad_v

    bar_h2 = int(h * 0.35)
    bar_w2 = 15
    bar_x2 = w - 130
    bar_y2 = ((h - bar_h2) // 2) - 130

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bar_x2 - 18, bar_y2 - 14), (bar_x2 + bar_w2 + 60, bar_y2 + bar_h2 + 14), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.35, frame, 0.65, 0, frame)

    cv2.rectangle(frame, (bar_x2, bar_y2), (bar_x2 + bar_w2, bar_y2 + bar_h2), (255,255,255), 2, cv2.LINE_AA)

    def val_to_y(v):
        v = max(min(v, a_max), a_min)
        ratio = (v - a_min) / (a_max - a_min)
        return int(bar_y2 + bar_h2 - ratio * bar_h2)

    def draw_marker(v, color, text):
        if math.isnan(v): return
        yy = val_to_y(v)
        cv2.line(frame, (bar_x2-6, yy), (bar_x2+bar_w2+6, yy), color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (bar_x2 - 100, yy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    draw_marker(cur_alt2, (255,100,0), "Container")
    draw_marker(cur_alt1, (0,200,255), "Payload")

    for frac in [1.0, 0.5, 0.0]:
        v = a_min + frac*(a_max - a_min)
        yy = val_to_y(v)
        cv2.line(frame, (bar_x2-4, yy), (bar_x2, yy), (255,255,255), 1, cv2.LINE_AA)
        lbl = f"{v:.1f}"
        cv2.putText(frame, lbl, (bar_x2 + bar_w2 + 8, yy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # ---------- BOTTOM LEFT: Attitude indicator ----------
    x0, y0, x1, y1 = lb_box
    radius = 65
    cx_ai = x0 + radius
    cy_ai = y0 - radius - 30
    pitch = to_float_or_nan(pitch_val)
    roll  = to_float_or_nan(roll_val)
    draw_attitude_indicator(frame, int(cx_ai), int(cy_ai), radius, pitch, roll)

    # ---------- WIDE YAW BAND IN THE MIDDLE ----------
    yaw = to_float_or_nan(yaw_val)
    y_bottom_offset = 30
    tape_h = 28
    draw_yaw_tape_center(frame, yaw_deg=yaw, y_bottom_offset=y_bottom_offset, width_ratio=0.6, tape_h=tape_h)

    # ---------- GPS MINI MAP ----------
    yaw_tape_top = h - y_bottom_offset - tape_h - 160   
    right_bar_x2 = bar_x2 + 100
    t_query = t_video - offset
    draw_gps_minimap(frame,
                     t_csv_all=t_csv_all,
                     lat_all=lat_all,
                     lon_all=lon_all,
                     t_query=t_query,
                     box_w=200, box_h=200,
                     margin=16,
                     right_bar_x=right_bar_x2,
                     yaw_tape_bottom=yaw_tape_top)

    # Debug row, bottom of the video
    dbg = f"Elapsed={t_video:6.2f}s  FPS={fps:.2f}  offset={offset:+.2f}s"
    (dbgw, dbgh), _ = cv2.getTextSize(dbg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    dx = (w - dbgw) // 2
    cv2.putText(frame, dbg, (dx, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, dbg, (dx, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# ---------- Numerical DF ----------
def build_numeric_df(df, colmap):
    cols = {}
    for k in NUMERIC_KEYS:
        c = colmap.get(k)
        if c and c in df.columns:
            cols[k] = normalize_numeric_series(df[c])
        else:
            cols[k] = pd.Series([np.nan]*len(df))
    return pd.DataFrame(cols)

def auto_map_columns(df):
    colmap = {}
    for key, cands in DEFAULT_COLS.items():
        colmap[key] = find_first_existing_col(df, cands)
    return colmap

# ---------- Arguments ----------
def parse_args():
    ap = argparse.ArgumentParser(description="CSV telemetry overlay on video")
    ap.add_argument("--video", required=True, help="Input video path (AVI/MP4)")
    ap.add_argument("--csv", required=True, help="Telemetry CSV path")
    ap.add_argument("--out", default="overlay_out.mp4", help="Output video path")
    ap.add_argument("--csv-time-col", default=None, help="CSV time column name (e.g., ‘Send Time’)")
    ap.add_argument("--csv-time-format", default=None, help="CSV time format (e.g., ‘%Y-%m-%d %H:%M:%S.%f’)")
    ap.add_argument("--video-start", default=None, help="The actual start time of the video (e.g., ‘2025-09-12 14:03:05.200’)")
    ap.add_argument("--packet-start", type=float, default=None, help="The packet number to be aligned with the t=0 moment of the video")
    ap.add_argument("--packet-dt", type=float, default=None, help="Inter-packet delay assumption (s), default 0.1")
    ap.add_argument("--sample-dt", type=float, default=None, help="When the information is not available, the line spacing (s)")
    ap.add_argument("--offset", type=float, default=0.0, help="CSV -> video time offset (s). CSV time + offset = video time")
    ap.add_argument("--no-preview", action="store_true", help="Render directly without opening the preview")
    ap.add_argument("--codec", default="mp4v", help="Four-character codec (mp4v, XVID, avc1, etc.)")
    return ap.parse_args()

# ---------- Render ----------
def render_video(cap, fps, frame_w, frame_h, t_csv, df_num, args, df_raw, colmap, alt_minmax):
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.out, fourcc, fps, (frame_w, frame_h))
    if not out.isOpened():
        print("Error: VideoWriter could not be opened. Try a different --codec (e.g., XVID, mp4v, avc1).")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Render starting: {total_frames} frame, {fps:.2f} FPS -> {total_frames/fps:.1f} second")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t_video = fi / fps
        t_csv_query = t_video - args.offset
        hud_num  = interpolate_row_at_time(t_csv, df_num, t_csv_query)
        hud_text = discrete_text_at_time(t_csv, df_raw, colmap, t_csv_query)
        hud = {**hud_num, **hud_text}
        draw_panel(frame, hud, t_video, fps, args.offset, alt_minmax=alt_minmax,
                   t_csv_all=t_csv,
                   lat_all=df_num["lat"].to_numpy() if "lat" in df_num.columns else None,
                   lon_all=df_num["lon"].to_numpy() if "lon" in df_num.columns else None)
        out.write(frame)
        if fi % int(max(1, fps*5)) == 0 and fi > 0:
            print(f"… {fi}/{total_frames} ({100.0*fi/total_frames:.1f}%)")

    out.release()
    print(f"Successful! Render completed: {args.out}")
    return True

# ---------- Main ----------
def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print("Error: Video not found!:", args.video); sys.exit(1)
    if not os.path.exists(args.csv):
        print("Error: CSV not found!:", args.csv); sys.exit(1)

    df = read_csv_flex(args.csv)
    colmap = auto_map_columns(df)
    df_num = build_numeric_df(df, colmap)

    t_csv = build_telemetry_timeseries(df, args, colmap)
    order = np.argsort(t_csv)
    t_csv = t_csv[order]
    df_num = df_num.iloc[order].reset_index(drop=True)
    df_sorted = df.iloc[order].reset_index(drop=True)

    # ALT bar ölçeği: dataset bazlı global min/max
    alt_candidates = []
    for key in ["alt", "alt1", "alt2"]:
        if key in df_num.columns:
            alt_candidates.append(df_num[key].to_numpy())
    if alt_candidates:
        all_alt = np.concatenate(alt_candidates)
        all_alt = all_alt[~np.isnan(all_alt)]
        if len(all_alt) > 0:
            a_min, a_max = float(np.min(all_alt)), float(np.max(all_alt))
            if abs(a_max - a_min) < 1e-6:
                a_min -= 1.0; a_max += 1.0
            pad = 0.05 * (a_max - a_min)
            alt_minmax = (a_min - pad, a_max + pad)
        else:
            alt_minmax = (0.0, 700.0)
    else:
        alt_minmax = (0.0, 700.0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: The video could not be opened!:", args.video); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.no_preview:
        render_video(cap, fps, frame_w, frame_h, t_csv, df_num, args, df_sorted, colmap, alt_minmax)
        cap.release()
        return

    print("Preview: j/l: ±0.1s, J/L: ±1.0s, s: render, q: output")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t_video = fi / fps
        t_csv_query = t_video - args.offset
        hud_num  = interpolate_row_at_time(t_csv, df_num, t_csv_query)
        hud_text = discrete_text_at_time(t_csv, df_sorted, colmap, t_csv_query)
        hud = {**hud_num, **hud_text}
        draw_panel(frame, hud, t_video, fps, args.offset, alt_minmax=alt_minmax,
                   t_csv_all=t_csv,
                   lat_all=df_num["lat"].to_numpy() if "lat" in df_num.columns else None,
                   lon_all=df_num["lon"].to_numpy() if "lon" in df_num.columns else None)
        cv2.imshow("Telemetry Overlay Preview by @ibrhmysil", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('j'):
            args.offset -= 0.1
        elif key == ord('l'):
            args.offset += 0.1
        elif key == ord('J'):
            args.offset -= 1.0
        elif key == ord('L'):
            args.offset += 1.0
        elif key == ord('s'):
            cv2.destroyWindow("Telemetry Overlay Preview by @ibrhmysil")
            render_video(cap, fps, frame_w, frame_h, t_csv, df_num, args, df_sorted, colmap, alt_minmax)
            cv2.namedWindow("Telemetry Overlay Preview by @ibrhmysil", cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()