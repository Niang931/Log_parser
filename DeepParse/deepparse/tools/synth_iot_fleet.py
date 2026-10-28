"""Synthetic IoT fleet log generator.

Generates IoT fleet logs for DeepParse instruction fine-tuning and evaluation.
Conforms to LogHub-2k / DeepParse dataset format:
  - raw.log: 2,000 log lines
  - templates.json: list of entries with {"cluster_id": <int>, "template": <str>}
  - manifest.json: dataset metadata and sha256 checksum

Ground-truth convention (gt_version 2), aligned with LogHub-2k / DeepParse:
  * A dash-joined identifier (``TH-75230``, ``cmd-4960``, ``gw-edge-53``,
    ``CAM-030``) is ONE variable and is masked as a whole: ``SN:<*>``,
    ``id=<*>``, ``Camera <*>``.  This mirrors how LogHub treats ``blk_123``.
  * A letter prefix glued directly to digits (``CPU0``, ``wlan0``) keeps the
    prefix: ``CPU<*>``.  Interfaces that never vary stay static.
  * Every ``<*>`` slot MUST take at least two distinct values in the
    generated corpus, and every value that varies MUST be a ``<*>`` slot.
    ``verify_ground_truth`` enforces this so no "fake constant" (a value the
    generator happens to emit identically every time) can leak into the
    templates and penalise a parser that correctly masks it.

Categories generated (2,000 lines):
  1. Telemetry Logs: 600 lines (30%)
  2. Network & Connection Logs: 400 lines (20%)
  3. Application & Firmware Logs: 350 lines (17.5%)
  4. Device Twin / Shadow Operations: 200 lines (10%)
  5. C2D / Control Logs: 150 lines (7.5%)
  6. Identity & Authentication Logs: 120 lines (6%)
  7. System & Kernel Logs: 100 lines (5%)
  8. OTA Update Logs: 80 lines (4%)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable
from pathlib import Path


def _rand_ts(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    ms = rng.randint(100, 999)

    return f"2026-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}.{ms:03d}"


def _fmt(ts: str, level: str, raw_msg: str, tpl_msg: str) -> tuple[str, str]:
    return f"{ts} [{level}] {raw_msg}", f"<*> [<*>] {tpl_msg}"


# _________ 1. Telemetry (600 lines) __________________________
def _gen_telemetry_camera_motion(rng: random.Random) -> tuple[str, str]:
    cam_id = f"{rng.randint(1, 99):03d}"
    region = rng.choice(["front_door", "driveway", "backyard", "garage", "hallway"])
    conf = rng.randint(70, 99)
    x1, y1 = rng.randint(10, 200), rng.randint(10, 200)
    x2, y2 = x1 + rng.randint(50, 200), y1 + rng.randint(50, 200)
    raw = f"Camera CAM-{cam_id} motion detected region={region} confidence={conf}% bbox=[{x1},{y1},{x2},{y2}]"
    tpl = "Camera <*> motion detected region=<*> confidence=<*>% bbox=[<*>,<*>,<*>,<*>]"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_camera_stream(rng: random.Random) -> tuple[str, str]:
    cam_id = f"{rng.randint(1, 99):03d}"
    stream = rng.choice(["main", "sub", "live"])
    fps = rng.choice([15.0, 24.0, 30.0, 60.0])
    bitrate = rng.choice([1024, 2048, 4096, 8192])
    dropped = rng.randint(0, 5)
    raw = f"Camera CAM-{cam_id} stream={stream} fps={fps:.1f} bitrate={bitrate}kbps dropped_frames={dropped}"
    tpl = "Camera <*> stream=<*> fps=<*> bitrate=<*>kbps dropped_frames=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_thermostat(rng: random.Random) -> tuple[str, str]:
    sn = f"TH-{rng.randint(10000, 99999)}"
    temp = round(rng.uniform(18.0, 32.0), 1)
    target = round(rng.uniform(20.0, 26.0), 1)
    mode = rng.choice(["auto", "heat", "cool", "eco"])
    hum = rng.randint(30, 85)
    raw = f"Thermostat SN:{sn} temp={temp:.1f}C target={target:.1f}C mode={mode} humidity={hum}%"
    tpl = "Thermostat SN:<*> temp=<*>C target=<*>C mode=<*> humidity=<*>%"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_battery(rng: random.Random) -> tuple[str, str]:
    sn = f"TH-{rng.randint(10000, 99999)}"
    battery = rng.randint(10, 100)
    source = rng.choice(["battery", "line_power", "solar"])
    raw = f"Thermostat SN:{sn} battery={battery}% power_source={source}"
    tpl = "Thermostat SN:<*> battery=<*>% power_source=<*>"
    level = "WARN" if battery < 20 else "INFO"
    return _fmt(_rand_ts(rng), level, raw, tpl)


def _gen_telemetry_smart_meter(rng: random.Random) -> tuple[str, str]:
    mtr = f"{rng.randint(1000, 9999)}"
    voltage = round(rng.uniform(215.0, 240.0), 1)
    current = round(rng.uniform(0.5, 25.0), 2)
    power = round(voltage * current, 1)
    pf = round(rng.uniform(0.85, 0.99), 2)
    raw = f"SmartMeter MTR-{mtr} voltage={voltage:.1f}V current={current:.2f}A active_power={power:.1f}W pf={pf:.2f}"
    tpl = "SmartMeter <*> voltage=<*>V current=<*>A active_power=<*>W pf=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_smart_meter_kwh(rng: random.Random) -> tuple[str, str]:
    mtr = f"{rng.randint(1000, 9999)}"
    kwh = round(rng.uniform(100.0, 5000.0), 1)
    freq = round(rng.uniform(49.8, 50.2), 1)
    status = rng.choice(["normal", "warning", "calibrated"])
    raw = f"SmartMeter MTR-{mtr} cumulative_kwh={kwh:.1f} frequency={freq:.1f}Hz status={status}"
    tpl = "SmartMeter <*> cumulative_kwh=<*> frequency=<*>Hz status=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_air_quality(rng: random.Random) -> tuple[str, str]:
    sensor = f"{rng.randint(1, 50):02d}"
    pm25 = round(rng.uniform(5.0, 85.0), 1)
    pm10 = round(rng.uniform(10.0, 150.0), 1)
    co2 = rng.randint(400, 1500)
    voc = rng.randint(50, 400)
    raw = f"AirSensor ENV-{sensor} pm25={pm25:.1f}ug/m3 pm10={pm10:.1f}ug/m3 co2={co2}ppm voc={voc}ppb"
    tpl = "AirSensor <*> pm25=<*>ug/m3 pm10=<*>ug/m3 co2=<*>ppm voc=<*>ppb"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_inverter(rng: random.Random) -> tuple[str, str]:
    inv = f"{rng.randint(1, 20):03d}"
    pv_v = round(rng.uniform(300.0, 600.0), 1)
    pv_p = rng.randint(500, 5000)
    grid_f = round(rng.uniform(49.9, 50.1), 1)
    yield_td = round(rng.uniform(1.0, 35.0), 1)
    raw = f"Inverter INV-{inv} pv_voltage={pv_v:.1f}V pv_power={pv_p}W grid_freq={grid_f:.1f}Hz yield_today={yield_td:.1f}kWh"
    tpl = "Inverter <*> pv_voltage=<*>V pv_power=<*>W grid_freq=<*>Hz yield_today=<*>kWh"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_telemetry_tracker(rng: random.Random) -> tuple[str, str]:
    trk = f"{rng.randint(100, 999)}"
    lat = round(rng.uniform(10.0, 20.0), 4)
    lon = round(rng.uniform(105.0, 108.0), 4)
    speed = round(rng.uniform(0.0, 90.0), 1)
    sats = rng.randint(4, 12)
    raw = f"Tracker TRK-{trk} gps lat={lat:.4f} lon={lon:.4f} speed={speed:.1f}km/h satellites={sats}"
    tpl = "Tracker <*> gps lat=<*> lon=<*> speed=<*>km/h satellites=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


# _________ 2. Network & Connection (400 lines) __________________________
def _gen_network_wifi_connected(rng: random.Random) -> tuple[str, str]:
    ssid = rng.choice(["SmartFleet_5G", "IoT_Secure", "Warehouse_AP", "Factory_Guest", "Home_Mesh"])
    bssid = ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))
    ch = rng.choice([1, 6, 11, 36, 40, 44, 48])
    rssi = rng.randint(-85, -45)
    raw = f"WiFi interface wlan0 connected to SSID \"{ssid}\" bssid={bssid} channel={ch} rssi={rssi}dBm"
    tpl = "WiFi interface wlan0 connected to SSID \"<*>\" bssid=<*> channel=<*> rssi=<*>dBm"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_network_wifi_disconnected(rng: random.Random) -> tuple[str, str]:
    reason = rng.choice(["beacon_loss", "auth_expired", "ap_leave", "low_rssi"])
    code = rng.choice([2, 3, 4, 8, 15])
    raw = f"WiFi interface wlan0 disconnected reason=\"{reason}\" code={code}"
    tpl = "WiFi interface wlan0 disconnected reason=\"<*>\" code=<*>"
    return _fmt(_rand_ts(rng), "WARN", raw, tpl)


def _gen_network_cellular(rng: random.Random) -> tuple[str, str]:
    rat = rng.choice(["LTE", "NR", "WCDMA"])
    mcc_mnc = rng.choice(["452_04", "452_01", "310_410", "208_10"])
    rsrp = rng.randint(-115, -75)
    rsrq = rng.randint(-16, -7)
    cell_id = f"0x{rng.randint(0x1000, 0xFFFF):04X}"
    raw = f"Cellular modem rat={rat} mcc_mnc={mcc_mnc} rsrp={rsrp}dBm rsrq={rsrq}dB cell_id={cell_id}"
    tpl = "Cellular modem rat=<*> mcc_mnc=<*> rsrp=<*>dBm rsrq=<*>dB cell_id=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_network_dhcp_lease(rng: random.Random) -> tuple[str, str]:
    ip = f"192.168.1.{rng.randint(2, 250)}"
    gw = rng.choice(["192.168.1.1", "192.168.1.254"])
    mask = rng.choice(["255.255.255.0", "255.255.0.0"])
    dns = rng.choice(["8.8.8.8", "1.1.1.1", "192.168.1.1"])
    raw = f"DHCP client lease acquired ip={ip} netmask={mask} gateway={gw} dns={dns}"
    tpl = "DHCP client lease acquired ip=<*> netmask=<*> gateway=<*> dns=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_network_dhcp_renew(rng: random.Random) -> tuple[str, str]:
    iface = rng.choice(["eth0", "wlan0", "wwan0"])
    xid = f"0x{rng.randint(0x10000, 0xFFFFF):05X}"
    raw = f"DHCP client renewing lease for interface {iface} xid={xid}"
    tpl = "DHCP client renewing lease for interface <*> xid=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_network_mqtt_connect(rng: random.Random) -> tuple[str, str]:
    cid = f"gw-edge-{rng.randint(10, 99)}"
    broker_ip = f"192.168.1.{rng.randint(1, 10)}"
    port = rng.choice([1883, 8883])
    ka = rng.choice([30, 60, 120])
    raw = f"MQTT client id={cid} connected to broker {broker_ip}:{port} keepalive={ka}s"
    tpl = "MQTT client id=<*> connected to broker <*> keepalive=<*>s"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_network_mqtt_ping(rng: random.Random) -> tuple[str, str]:
    cid = f"gw-edge-{rng.randint(10, 99)}"
    rtt = rng.randint(8, 65)
    raw = f"MQTT client id={cid} pingreq sent awaiting pingresp rtt={rtt}ms"
    tpl = "MQTT client id=<*> pingreq sent awaiting pingresp rtt=<*>ms"
    return _fmt(_rand_ts(rng), "DEBUG", raw, tpl)


def _gen_network_mqtt_disconnect(rng: random.Random) -> tuple[str, str]:
    cid = f"gw-edge-{rng.randint(10, 99)}"
    clean = rng.choice(["true", "false"])
    rc = 0 if clean == "true" else rng.choice([1, 2, 7, 16])
    raw = f"MQTT client id={cid} disconnected clean={clean} rc={rc}"
    tpl = "MQTT client id=<*> disconnected clean=<*> rc=<*>"
    return _fmt(_rand_ts(rng), "WARN", raw, tpl)


def _gen_network_socket_err(rng: random.Random) -> tuple[str, str]:
    err = rng.choice([104, 110, 111, 113])
    fd = rng.randint(3, 32)
    rem_ip = f"10.0.0.{rng.randint(1, 50)}"
    rem_port = rng.choice([443, 8883, 5000])
    raw = f"Network socket err={err} on fd={fd} remote={rem_ip}:{rem_port}"
    tpl = "Network socket err=<*> on fd=<*> remote=<*>"
    return _fmt(_rand_ts(rng), "ERROR", raw, tpl)


# _________ 3. Application & Firmware (350 lines) __________________________
def _gen_app_daemon_started(rng: random.Random) -> tuple[str, str]:
    svc = rng.choice(["video_streamer", "sensor_collector", "ota_agent", "telemetry_worker"])
    pid = rng.randint(200, 9999)
    mem = rng.choice([64, 128, 256, 512])
    raw = f"AppDaemon service={svc} pid={pid} started with memory_limit={mem}MB"
    tpl = "AppDaemon service=<*> pid=<*> started with memory_limit=<*>MB"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_app_storage_flash(rng: random.Random) -> tuple[str, str]:
    part = rng.choice(["/dev/mtdblock2", "/dev/mtdblock3", "/dev/mtdblock5"])
    blocks = rng.choice([16, 32, 64, 128])
    dur = rng.randint(5, 45)
    st = rng.choice(["OK", "VERIFIED"])
    raw = f"Storage flash partition={part} write_blocks={blocks} duration={dur}ms status={st}"
    tpl = "Storage flash partition=<*> write_blocks=<*> duration=<*>ms status=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_app_buffer_pool(rng: random.Random) -> tuple[str, str]:
    max_b = rng.choice([32768, 65536])
    used_b = rng.randint(4096, 30000)
    wm = int(used_b / max_b * 100)
    flushed = rng.randint(2, 30)
    raw = f"Buffer pool usage={used_b}/{max_b} bytes watermark={wm}% flushed={flushed} items"
    tpl = "Buffer pool usage=<*>/<*> bytes watermark=<*>% flushed=<*> items"
    return _fmt(_rand_ts(rng), "DEBUG", raw, tpl)


def _gen_app_camera_rtsp(rng: random.Random) -> tuple[str, str]:
    sess_id = f"sess-{rng.randint(100, 999)}"
    client_ip = f"192.168.1.{rng.randint(20, 99)}"
    port = rng.choice([554, 8554])
    codec = rng.choice(["H264", "H265"])
    res = rng.choice(["1280x720", "1920x1080", "2560x1440"])
    raw = f"CameraDaemon RTSP session id={sess_id} client={client_ip}:{port} codec={codec} resolution={res}"
    tpl = "CameraDaemon RTSP session id=<*> client=<*> codec=<*> resolution=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_app_watchdog(rng: random.Random) -> tuple[str, str]:
    svc = rng.choice(["telemetry_worker", "network_monitor", "camera_pipeline"])
    deadline = rng.choice([3000, 5000, 10000])
    raw = f"Watchdog timer kicked by service={svc} next_deadline={deadline}ms"
    tpl = "Watchdog timer kicked by service=<*> next_deadline=<*>ms"
    return _fmt(_rand_ts(rng), "DEBUG", raw, tpl)


def _gen_app_memory_heap(rng: random.Random) -> tuple[str, str]:
    free_kb = rng.randint(800, 4000)
    alloc_kb = rng.randint(1000, 8000)
    peak_kb = alloc_kb + rng.randint(200, 1500)
    frag = rng.randint(1, 15)
    raw = f"Memory heap free={free_kb}KB alloc={alloc_kb}KB peak={peak_kb}KB fragmentation={frag}%"
    tpl = "Memory heap free=<*>KB alloc=<*>KB peak=<*>KB fragmentation=<*>%"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


# _________ 4. Device Twin / Shadow Operations (200 lines) __________________________
def _gen_twin_reported(rng: random.Random) -> tuple[str, str]:
    key = rng.choice(["firmware_ver", "motion_sensitivity", "reporting_interval", "led_status"])
    val = rng.choice(["v2.1.0", "medium", "60", "disabled", "auto"])
    ver = rng.randint(10, 150)
    raw = f"DeviceTwin reported state updated key=\"{key}\" value=\"{val}\" version={ver}"
    tpl = "DeviceTwin reported state updated key=\"<*>\" value=\"<*>\" version=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_twin_desired_delta(rng: random.Random) -> tuple[str, str]:
    prop = rng.choice(["target_temp", "reporting_rate", "stream_profile", "night_mode"])
    val = rng.choice(["23.5", "10", "high_def", "on", "auto"])
    exp_ver = rng.randint(10, 150)
    raw = f"DeviceTwin desired delta received property=\"{prop}\" desired_value=\"{val}\" expected_ver={exp_ver}"
    tpl = "DeviceTwin desired delta received property=\"<*>\" desired_value=\"<*>\" expected_ver=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_twin_sync(rng: random.Random) -> tuple[str, str]:
    dur = rng.randint(40, 350)
    ver = rng.randint(10, 150)
    status = rng.choice(["SYNCED", "PARTIAL"])
    raw = f"DeviceTwin sync completed in {dur}ms status={status} local_ver={ver} cloud_ver={ver}"
    tpl = "DeviceTwin sync completed in <*>ms status=<*> local_ver=<*> cloud_ver=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_twin_rejected(rng: random.Random) -> tuple[str, str]:
    reason = rng.choice(["version_conflict", "schema_mismatch", "payload_too_large"])
    code = rng.choice([400, 409, 413])
    raw = f"DeviceTwin patch rejected reason=\"{reason}\" error_code={code}"
    tpl = "DeviceTwin patch rejected reason=\"<*>\" error_code=<*>"
    return _fmt(_rand_ts(rng), "WARN", raw, tpl)


# _________ 5. C2D / Control Logs (150 lines) __________________________
def _gen_c2d_command_rcvd(rng: random.Random) -> tuple[str, str]:
    cmd_id = f"cmd-{rng.randint(1000, 9999)}"
    method = rng.choice(["unlock_door", "reboot_device", "trigger_alarm", "capture_snapshot"])
    payload_sz = rng.choice([16, 32, 64, 128, 256])
    raw = f"C2D command id={cmd_id} method=\"{method}\" received payload_size={payload_sz}B"
    tpl = "C2D command id=<*> method=\"<*>\" received payload_size=<*>B"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_c2d_actuator_exec(rng: random.Random) -> tuple[str, str]:
    dl_id = f"{rng.randint(1000, 9999)}"
    action = rng.choice(["unlock", "lock", "calibrate_motor"])
    user = rng.choice(["alice", "bob", "cloud_admin", "scheduler"])
    result = rng.choice(["SUCCESS", "FAILED"])
    raw = f"Actuator DL-{dl_id} executed action=\"{action}\" user=\"{user}\" result={result}"
    tpl = "Actuator <*> executed action=\"<*>\" user=\"<*>\" result=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_c2d_ptz_move(rng: random.Random) -> tuple[str, str]:
    cam = f"{rng.randint(1, 20):03d}"
    pan = round(rng.uniform(0.0, 359.0), 1)
    tilt = round(rng.uniform(-20.0, 90.0), 1)
    zoom = round(rng.uniform(1.0, 5.0), 1)
    raw = f"PTZ controller moved cam_id=CAM-{cam} pan={pan:.1f} tilt={tilt:.1f} zoom={zoom:.1f}x"
    tpl = "PTZ controller moved cam_id=<*> pan=<*> tilt=<*> zoom=<*>x"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_c2d_relay_switch(rng: random.Random) -> tuple[str, str]:
    sw = rng.choice(["heater_relay", "cooling_fan", "main_valve", "irrigation_pump"])
    st = rng.choice(["on", "off"])
    dur = rng.choice([0, 30, 60, 300])
    raw = f"Relay controller switch={sw} set to state={st} duration={dur}s"
    tpl = "Relay controller switch=<*> set to state=<*> duration=<*>s"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_c2d_response_sent(rng: random.Random) -> tuple[str, str]:
    cmd_id = f"cmd-{rng.randint(1000, 9999)}"
    code = rng.choice([200, 202, 204])
    exec_t = rng.randint(12, 120)
    raw = f"C2D response sent for command id={cmd_id} status_code={code} exec_time={exec_t}ms"
    tpl = "C2D response sent for command id=<*> status_code=<*> exec_time=<*>ms"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


# _________ 6. Identity & Authentication Logs (120 lines) __________________________
def _gen_auth_tls_handshake(rng: random.Random) -> tuple[str, str]:
    cipher = rng.choice([
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "TLS_AES_128_GCM_SHA256",
    ])
    cert_sn = f"0x{rng.randint(0x100000, 0xFFFFFF):06X}"
    raw = f"AuthManager TLS handshake succeeded cipher={cipher} cert_sn={cert_sn}"
    tpl = "AuthManager TLS handshake succeeded cipher=<*> cert_sn=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_auth_token_refresh(rng: random.Random) -> tuple[str, str]:
    scope = rng.choice(["device:telemetry", "device:twin", "device:c2d", "fleet:all"])
    expires = rng.choice([1800, 3600, 7200])
    raw = f"AuthManager token refreshed scope=\"{scope}\" expires_in={expires}s"
    tpl = "AuthManager token refreshed scope=\"<*>\" expires_in=<*>s"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_auth_keypad_accept(rng: random.Random) -> tuple[str, str]:
    dl = f"{rng.randint(1000, 9999)}"
    user = rng.choice(["alice", "bob", "guest_01", "cleaner"])
    slot = rng.randint(1, 10)
    raw = f"Keypad lock=DL-{dl} code accepted user=\"{user}\" slot={slot}"
    tpl = "Keypad lock=<*> code accepted user=\"<*>\" slot=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_auth_keypad_reject(rng: random.Random) -> tuple[str, str]:
    dl = f"{rng.randint(1000, 9999)}"
    max_att = rng.choice([3, 5])
    attempt = rng.randint(1, max_att)
    lockout = 60 if attempt >= 3 else 0
    raw = f"Keypad lock=DL-{dl} code rejected attempt={attempt}/{max_att} lockout={lockout}s"
    tpl = "Keypad lock=<*> code rejected attempt=<*>/<*> lockout=<*>s"
    return _fmt(_rand_ts(rng), "WARN", raw, tpl)


def _gen_auth_rfid_scan(rng: random.Random) -> tuple[str, str]:
    tag = f"E280116060{rng.randint(100000, 999999)}"
    door = rng.choice(["main_gate", "server_room", "warehouse_entry"])
    access = rng.choice(["GRANTED", "DENIED"])
    raw = f"RFID reader scanned tag_id={tag} door=\"{door}\" access={access}"
    tpl = "RFID reader scanned tag_id=<*> door=\"<*>\" access=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


# _________ 7. System & Kernel Logs (100 lines) __________________________
def _gen_sys_kernel_boot(rng: random.Random) -> tuple[str, str]:
    kver = rng.choice(["5.15.76-rt", "5.10.120-v7+", "6.1.21-iot"])
    gcc = rng.choice(["10.2.1", "11.2.0", "12.2.0"])
    raw = f"Kernel: Linux version {kver} (gcc version {gcc}) armv7l SMP"
    tpl = "Kernel: Linux version <*> (gcc version <*>) armv7l SMP"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_sys_thermal_throttle(rng: random.Random) -> tuple[str, str]:
    cpu = rng.choice([0, 1, 2, 3])
    t = round(rng.uniform(75.0, 84.5), 1)
    max_t = rng.choice([85.0, 90.0, 95.0])
    raw = f"Kernel: CPU{cpu} thermal throttling event temp={t:.1f}C max={max_t:.1f}C"
    tpl = "Kernel: CPU<*> thermal throttling event temp=<*>C max=<*>C"
    return _fmt(_rand_ts(rng), "WARN", raw, tpl)


def _gen_sys_udev_device(rng: random.Random) -> tuple[str, str]:
    subsys = rng.choice(["usb", "net", "mmc", "tty"])
    dev = rng.choice([
        "/devices/platform/soc/usb1",
        "/devices/platform/soc/ttyAMA0",
        "/devices/virtual/net/wlan0",
    ])
    raw = f"Kernel: udev device added subsystem={subsys} devpath={dev}"
    tpl = "Kernel: udev device added subsystem=<*> devpath=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_sys_mount_vfs(rng: random.Random) -> tuple[str, str]:
    fs = rng.choice(["ext4", "squashfs", "f2fs"])
    dev = rng.choice(["/dev/mmcblk0p2", "/dev/mtdblock4", "/dev/sda1"])
    raw = f"Kernel: VFS: Mounted root filesystem {fs} on device {dev} ro"
    tpl = "Kernel: VFS: Mounted root filesystem <*> on device <*> ro"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_sys_eth_link(rng: random.Random) -> tuple[str, str]:
    speed = rng.choice([10, 100, 1000])
    flow = rng.choice(["rx/tx", "rx", "tx", "off"])
    raw = f"Kernel: eth0: link up {speed}Mbps full-duplex flow-control={flow}"
    tpl = "Kernel: eth0: link up <*>Mbps full-duplex flow-control=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


# _________ 8. OTA Update Logs (80 lines) __________________________
def _gen_ota_check(rng: random.Random) -> tuple[str, str]:
    v = f"v{rng.randint(2, 3)}.{rng.randint(0, 9)}.{rng.randint(0, 5)}"
    sz = rng.randint(8000000, 32000000)
    url = f"https://ota.iot.io/fw/{v}.bin"
    raw = f"OTA check: update found target_ver=\"{v}\" size={sz}B url=\"{url}\""
    tpl = "OTA check: update found target_ver=\"<*>\" size=<*>B url=\"<*>\""
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_ota_progress(rng: random.Random) -> tuple[str, str]:
    total = rng.choice([50, 100, 200])
    chunk = rng.randint(1, total)
    pct = int(chunk * 100 / total)
    bytes_dl = int(15482880 * (chunk / total))
    raw = f"OTA download progress chunk={chunk}/{total} bytes={bytes_dl} percent={pct}%"
    tpl = "OTA download progress chunk=<*>/<*> bytes=<*> percent=<*>%"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_ota_verify(rng: random.Random) -> tuple[str, str]:
    sha = "".join(f"{rng.randint(0, 255):02x}" for _ in range(32))
    result = rng.choice(["VALID", "INVALID"])
    raw = f"OTA verify payload sha256={sha} result={result}"
    tpl = "OTA verify payload sha256=<*> result=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_ota_flash(rng: random.Random) -> tuple[str, str]:
    part = rng.choice(["bank_a", "bank_b"])
    offset = f"0x{rng.randint(0x100000, 0x800000):08X}"
    ln = rng.choice([8388608, 15482880, 20971520])
    raw = f"OTA flash writing partition=\"{part}\" offset={offset} length={ln}"
    tpl = "OTA flash writing partition=\"<*>\" offset=<*> length=<*>"
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


def _gen_ota_reboot(rng: random.Random) -> tuple[str, str]:
    part = rng.choice(["bank_a", "bank_b"])
    raw = f"OTA reboot scheduled to switch active bank to partition=\"{part}\""
    tpl = "OTA reboot scheduled to switch active bank to partition=\"<*>\""
    return _fmt(_rand_ts(rng), "INFO", raw, tpl)


_GENERATORS: list[tuple[int, list[Callable[[random.Random], tuple[str, str]]]]] = [
    (600, [
        _gen_telemetry_camera_motion,
        _gen_telemetry_camera_stream,
        _gen_telemetry_thermostat,
        _gen_telemetry_battery,
        _gen_telemetry_smart_meter,
        _gen_telemetry_smart_meter_kwh,
        _gen_telemetry_air_quality,
        _gen_telemetry_inverter,
        _gen_telemetry_tracker,
    ]),
    (400, [
        _gen_network_wifi_connected,
        _gen_network_wifi_disconnected,
        _gen_network_cellular,
        _gen_network_dhcp_lease,
        _gen_network_dhcp_renew,
        _gen_network_mqtt_connect,
        _gen_network_mqtt_ping,
        _gen_network_mqtt_disconnect,
        _gen_network_socket_err,
    ]),
    (350, [
        _gen_app_daemon_started,
        _gen_app_storage_flash,
        _gen_app_buffer_pool,
        _gen_app_camera_rtsp,
        _gen_app_watchdog,
        _gen_app_memory_heap,
    ]),
    (200, [
        _gen_twin_reported,
        _gen_twin_desired_delta,
        _gen_twin_sync,
        _gen_twin_rejected,
    ]),
    (150, [
        _gen_c2d_command_rcvd,
        _gen_c2d_actuator_exec,
        _gen_c2d_ptz_move,
        _gen_c2d_relay_switch,
        _gen_c2d_response_sent,
    ]),
    (120, [
        _gen_auth_tls_handshake,
        _gen_auth_token_refresh,
        _gen_auth_keypad_accept,
        _gen_auth_keypad_reject,
        _gen_auth_rfid_scan,
    ]),
    (100, [
        _gen_sys_kernel_boot,
        _gen_sys_thermal_throttle,
        _gen_sys_udev_device,
        _gen_sys_mount_vfs,
        _gen_sys_eth_link,
    ]),
    (80, [
        _gen_ota_check,
        _gen_ota_progress,
        _gen_ota_verify,
        _gen_ota_flash,
        _gen_ota_reboot,
    ]),
]


def generate_iot_fleet(count: int = 2000, seed: int = 42) -> tuple[list[str], list[dict]]:
    """Generate synthetic IoT logs and ground truth template entries."""
    rng = random.Random(seed)
    scale = count / 2000.0

    raw_pairs: list[tuple[str, str]] = []
    for base_n, pool in _GENERATORS:
        n = round(base_n * scale)

        for _ in range(n):
            gen = rng.choice(pool)
            raw_pairs.append(gen(rng))

    rng.shuffle(raw_pairs)

    logs: list[str] = []
    entries: list[dict] = []
    template_to_id: dict[str, int] = {}

    for i, (line, tpl) in enumerate(raw_pairs, start=1):
        logs.append(line)
        cid = template_to_id.setdefault(tpl, len(template_to_id))
        entries.append({"line_id": i, "cluster_id": cid, "template": tpl})

    return logs, entries


GT_VERSION = 2
_HEADER_SLOTS = 2  # ``<*> [<*>]`` prepended by ``_fmt``: timestamp and level


def _slot_values(line: str, template: str) -> list[str] | None:
    """Return the raw substring filled into each ``<*>`` slot, or ``None`` on mismatch."""
    parts = template.split("<*>")
    pattern = "^" + "(.+?)".join(re.escape(p) for p in parts) + "$"
    match = re.match(pattern, line)
    return list(match.groups()) if match else None


def verify_ground_truth(logs: list[str], entries: list[dict]) -> list[str]:
    """Check the generated corpus against the ground-truth convention.

    Returns a list of human-readable problems (empty when the corpus is clean):

    * a line that does not match its own template;
    * a ``<*>`` slot whose value never changes across the corpus (a "fake
      constant" that would unfairly penalise a parser for masking it).
    """
    problems: list[str] = []
    per_slot: dict[tuple[str, int], set[str]] = {}
    for line, entry in zip(logs, entries):
        tpl = entry["template"]
        values = _slot_values(line, tpl)
        if values is None:
            problems.append(f"line does not match its template: {line!r} vs {tpl!r}")
            continue
        for idx, value in enumerate(values):
            # The header slots (timestamp, level) come from ``_fmt`` and are
            # shared by every template, so they are checked corpus-wide: a
            # template that only ever logs at INFO is still a valid template.
            key = ("<header>", idx) if idx < _HEADER_SLOTS else (tpl, idx)
            per_slot.setdefault(key, set()).add(value)
    for (tpl, idx), values in sorted(per_slot.items()):
        if len(values) < 2:
            problems.append(
                f"slot #{idx} of {tpl!r} is constant ({next(iter(values))!r}); "
                "make the generator vary it or make it static in the template"
            )
    return problems


def write_dataset(out_dir: Path, logs: list[str], entries: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = "\n".join(logs) + "\n"
    (out_dir / "raw.log").write_text(raw_text, encoding="utf-8")
    
    (out_dir / "templates.json").write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")

    manifest = {
        "name": out_dir.name,
        "logs": len(logs),
        "templates": len({e["cluster_id"] for e in entries}),
        "checksum": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "source": "deepparse/tools/synth_iot_fleet.py (synthetic IoT fleet)",
        "gt_version": GT_VERSION,
        "gt_convention": (
            "composite dash-joined ids are one <*>; letter+digit prefixes (CPU0) keep "
            "the prefix; every <*> slot varies in the corpus"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000, help="Number of log lines to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/data/IoT_fleet"),
        help="Destination directory for raw.log and templates.json",
    )
    args = parser.parse_args(argv)

    print(f"[synth_iot_fleet] Generating {args.count} IoT fleet log lines (seed={args.seed})...")
    logs, entries = generate_iot_fleet(count=args.count, seed=args.seed)
    problems = verify_ground_truth(logs, entries)
    if problems:
        for problem in problems:
            print(f"[synth_iot_fleet] GT problem: {problem}")
        return 1
    write_dataset(args.out_dir, logs, entries)
    print(
        f"[synth_iot_fleet] Successfully wrote {len(logs)} logs ({len({e['cluster_id'] for e in entries})} templates) "
        f"to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

