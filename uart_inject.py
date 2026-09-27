#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uart_inject.py -- Meltalice / RHEOX
====================================
Injecte les segments de test_vectors.npz vers le firmware STM32 via UART,
compare les sorties embarquées aux sorties Python de référence.

Protocole UART PC→STM32 :
  En-tête  : 4 octets magic [0xDE 0xAD 0xBE 0xEF]
  Segment  : N_SEG × float32 (little-endian), N_SEG = 4096
  CRC      : 2 octets CRC-16/CCITT sur le segment

Protocole STM32→PC (réponse, 12 octets) :
  z_max    : float32 (4 octets)
  v_star   : uint8   (1 octet, 1-indexé)
  alarm    : uint8   (1 octet, 0 ou 1)
  rho_pred : float32 (4 octets)   ← NB: NOT e_obs0 (offset 6 = rho_pred)
  CRC      : uint16  (2 octets)

Critère V1 :
  |z_max_C - z_max_Python| / max(|z_max_Python|, 1e-3) < 0.5%
  v_star_C == v_star_Python pour tous les segments

Usage :
  python uart_inject.py --port COM4 --vectors test_vectors.npz
"""

import argparse
import struct
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("⚠  pyserial non installé : pip install pyserial")

MAGIC     = bytes([0xDE, 0xAD, 0xBE, 0xEF])
BAUD      = 921600
TIMEOUT_S = 2.0

RESP_SIZE = 4 + 1 + 1 + 4 + 2       # z_max + v* + alarm + rho_pred + CRC
ZMAX_TOL  = 0.005                    # 0.5 % tolérance relative
VSTAR_EXACT = True


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def build_request(seg: np.ndarray) -> bytes:
    seg_f32 = seg.astype(np.float32)
    payload = seg_f32.tobytes()
    crc = crc16_ccitt(payload)
    return MAGIC + payload + struct.pack("<H", crc)


def parse_response(raw: bytes):
    if len(raw) != RESP_SIZE:
        raise ValueError(f"longueur réponse inattendue : {len(raw)} (attendu {RESP_SIZE})")
    z_max,   = struct.unpack_from("<f", raw, 0)
    v_star   = raw[4]
    alarm    = bool(raw[5])
    rho_pred = struct.unpack_from("<f", raw, 6)[0]
    crc_recv = struct.unpack_from("<H", raw, 10)[0]
    crc_calc = crc16_ccitt(raw[:10])
    if crc_recv != crc_calc:
        raise ValueError(f"CRC réponse invalide : reçu={crc_recv:#06x} calc={crc_calc:#06x}")
    return z_max, v_star, alarm, rho_pred


def compare(idx, zmax_py, vstar_py, zmax_c, vstar_c, verbose=False):
    denom = max(abs(float(zmax_py)), 1e-3)
    err_rel = abs(float(zmax_c) - float(zmax_py)) / denom
    vstar_ok = (int(vstar_c) == int(vstar_py))
    segment_pass = (err_rel < ZMAX_TOL) and (not VSTAR_EXACT or vstar_ok)

    if verbose or not segment_pass:
        status = "✓" if segment_pass else "✗"
        print(f"  [{idx:4d}] {status}  z_max py={zmax_py:8.3f} C={zmax_c:8.3f} "
              f"err={err_rel*100:.3f}%  v* py={vstar_py} C={vstar_c} "
              f"{'OK' if vstar_ok else 'FAIL'}")
    return segment_pass, err_rel, vstar_ok


def run(port, vectors_path, verbose, dry_run):
    data = np.load(vectors_path)
    segments  = data["segments"]
    zmax_ref  = data["zmax"]
    vstar_ref = data["vstar"]
    N = segments.shape[1]
    n = len(segments)

    print(f"\n{'='*65}")
    print(f"  UART INJECT — {n} segments, N={N}, depuis {vectors_path}")
    print(f"  Port={port}  Baud={BAUD}  Tolérance z_max={ZMAX_TOL*100:.1f}%")
    print(f"{'='*65}\n")

    if dry_run:
        req = build_request(segments[0])
        print(f"  [DRY RUN] taille requête : {len(req)} octets "
              f"(attendu {4 + N*4 + 2})")
        print(f"  CRC seg[0] : {crc16_ccitt(segments[0].astype(np.float32).tobytes()):#06x}")
        print("  ✓ Protocole OK — brancher la carte pour le vrai test\n")
        return

    if not HAS_SERIAL:
        print("  ✗ pyserial manquant")
        return

    results = []
    with serial.Serial(port, BAUD, timeout=TIMEOUT_S) as ser:
        ser.reset_input_buffer()
        time.sleep(0.1)
        for i in range(n):
            ser.write(build_request(segments[i]))
            raw = ser.read(RESP_SIZE)

            if len(raw) == 0:
                print(f"  [{i:4d}] TIMEOUT — pas de réponse")
                results.append(False)
                continue
            try:
                zc, vc, ac, rp = parse_response(raw)
            except ValueError as e:
                print(f"  [{i:4d}] ERREUR PROTOCOLE : {e}")
                results.append(False)
                continue

            ok, err_rel, vs_ok = compare(i, zmax_ref[i], vstar_ref[i], zc, vc, verbose=verbose)
            results.append(ok)

            if (i + 1) % 50 == 0:
                print(f"  --- {i+1}/{n} traités  pass={100*sum(results)/len(results):.1f}% ---")

    n_pass = sum(results)
    pct = 100 * n_pass / n
    print(f"\n{'='*65}")
    print(f"  RÉSULTAT FINAL : {n_pass}/{n} PASS  ({pct:.2f}%)")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Injecte test_vectors.npz dans le firmware STM32 via UART")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--vectors", default="test_vectors.npz")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.port, args.vectors, args.verbose, args.dry_run)


if __name__ == "__main__":
    main()
