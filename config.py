"""Room layout and which ESP32 receivers are in use. Edit here only.

Origin = the corner with no board, x/y along the walls, z = height above the floor (metres).
"""
ROOM_SIZE_M = (2.5, 2.5, 2.5)

PHONE_POS_M = (1.25, 1.25, 0.76)  # iPhone 15 Pro hotspot = the only transmitter, room centre

# Active receivers: two ESP32-WROOM-32 DevKit V1 boards in two ADJACENT corners,
# ~0.2 m in from both walls, 40 cm high. Seen from the phone the two links are 90 deg apart.
RECEIVERS_M = {
    2: (2.30, 0.20, 0.40),
    3: (2.30, 2.30, 0.40),
}
ACTIVE_NODES = tuple(sorted(RECEIVERS_M))
BOARD = "ESP32-WROOM-32 DevKit V1 (classic ESP32, CP2102, micro-USB)"
