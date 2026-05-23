# LonelyScreen Integration Design

**Date:** 2026-05-22  
**Status:** Approved

## Goal

Replace the UxPlay AirPlay receiver with LonelyScreen as the screen mirroring source for the Block Blast Solver on Windows.

## Context

The solver captures a named PC window using `pygetwindow` + `mss`. The window title is configured in `config.py`. The user has installed LonelyScreen and successfully mirrored their iPhone 15 Pro screen to it.

## Changes

### 1. `block_blast_solver/config.py`
- Change `WINDOW_TITLE` from `"UxPlay"` to `"LonelyScreen AirPlay Receiver"`

### 2. `calibration_data.json` (if present)
- Delete the file. The old calibration ROI ratios were mapped to UxPlay's window dimensions and are invalid for LonelyScreen's different layout.
- On next run, the solver will detect missing calibration and prompt the user to re-draw ROIs.

## No Other Changes Required

The vision, solver, and visualizer modules are window-agnostic. Only the window title lookup is affected.

## First-Run Flow After Change

1. Open LonelyScreen on PC
2. On iPhone: Control Center → Screen Mirroring → LonelyScreen
3. Open Block Blast on iPhone
4. Run `main.py` → calibration UI opens
5. Draw ROI around the 8×8 board, then the 3 pieces area
6. Calibration saves to `calibration_data.json`
7. Solver runs normally on subsequent launches
