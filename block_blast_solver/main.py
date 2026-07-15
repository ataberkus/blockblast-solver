import cv2
import logging
import numpy as np
import time

from block_blast_solver import config
from block_blast_solver.modules import solver, vision
from block_blast_solver.modules.capture import WindowCapture
from block_blast_solver.modules.visualizer import Visualizer
from block_blast_solver.state import AppState

logger = logging.getLogger(__name__)
HUD_TITLE = "Block Blast AI Solver HUD"

# =====================================================================
# BLOCK BLAST SOLVER - ANA ORKESTRASYON VE PROGRAM DÖNGÜSÜ (main.py)
# =====================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting Block Blast Solver")

    config.load_calibration()
    try:
        capture = WindowCapture(config.WINDOW_TITLE)
    except RuntimeError as error:
        logger.error("%s", error)
        return

    visualizer = Visualizer()
    state = AppState()

    if config.BOARD_ROI is None or config.PIECES_ROI is None:
        logger.warning("Calibration is missing or invalid")
        if not capture.calibrate():
            logger.error("Calibration did not complete")
            capture.close()
            return

    logger.info("Solver loop active; press 'q' to quit or 'c' to recalibrate")
    cv2.namedWindow(HUD_TITLE, cv2.WINDOW_NORMAL)

    try:
        while True:
            loop_started = time.perf_counter()
            frame = capture.capture_frame()
            if frame is None:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                continue

            _, _, board_width, board_height = capture.get_roi_rect(config.BOARD_ROI, frame.shape)
            detected_board, occluded = vision.get_board_state(frame)
            detected_pieces = vision.get_pieces(frame, board_width / 8.0, board_height / 8.0)
            update = state.observe(
                detected_board,
                detected_pieces,
                config.DETECTION_STABLE_FRAMES,
                occluded,
            )

            board = state.last_board
            pieces = state.last_pieces
            if update is not None:
                board = update.board
                pieces = update.pieces
                if update.changed:
                    moves = None
                    score = 0.0
                    diagnostics = None
                    if any(piece is not None for piece in pieces):
                        solver_started = time.perf_counter()
                        moves, score, diagnostics = solver.solve_with_diagnostics(board, pieces)
                        solver_duration_ms = (time.perf_counter() - solver_started) * 1000.0
                        logger.info("Solution computed in %.1f ms (score %.1f)", solver_duration_ms, score)
                    state.accept(board, pieces, moves, score, diagnostics)

            hud_frame = visualizer.draw_hud(
                frame,
                board,
                pieces,
                state.last_moves,
                state.last_score,
                occluded,
                state.last_diagnostics,
            )
            cv2.imshow(HUD_TITLE, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                cv2.destroyWindow(HUD_TITLE)
                if capture.calibrate():
                    state.reset()
                cv2.namedWindow(HUD_TITLE, cv2.WINDOW_NORMAL)

            logger.debug("Loop completed in %.1f ms", (time.perf_counter() - loop_started) * 1000.0)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        capture.close()
        cv2.destroyAllWindows()
        logger.info("Stopped")

if __name__ == "__main__":
    main()
