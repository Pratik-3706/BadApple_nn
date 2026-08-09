"""
FastAPI WebSocket server for live Bad Apple streaming.

This is the bridge between the inference engine (Python/PyTorch) and
the browser (JS/Canvas/SVG). It runs the model, captures activations,
and pushes everything over a WebSocket as JSON.

Usage:
    python src/server.py
    -> open http://localhost:8000 in your browser
"""

import asyncio
import base64
import json
import os
import sys
import time
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))
from inference_engine import InferenceEngine


# ============================================================
# server config
# ============================================================
HOST = '127.0.0.1'
PORT = 8000
DEFAULT_FPS = 29.9

STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'static')
AUDIO_PATH = os.path.join(os.path.dirname(__file__), '..', 'bad_apple_vid', 'audio.mp3')
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'bad_apple_vid', 'vid.mp4')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

app = FastAPI(title="BadApple_X_NN")

# we load both best and latest engines at startup (if both exist).
# the latest checkpoint may not exist if training hasn't saved one yet.
engine_best = None
engine_latest = None


def _checkpoint_path(name):
    return os.path.join(CHECKPOINT_DIR, name)


@app.on_event("startup")
async def startup():
    global engine_best, engine_latest

    best_path = _checkpoint_path('badapple_nn.pt')
    latest_path = _checkpoint_path('badapple_nn_latest.pt')

    # always load best
    print("loading best checkpoint...")
    engine_best = InferenceEngine(checkpoint_path=best_path)

    # try loading latest too - it might not exist yet
    if os.path.exists(latest_path):
        print("loading latest checkpoint...")
        engine_latest = InferenceEngine(checkpoint_path=latest_path)
    else:
        print("no latest checkpoint found, comparison mode will use best for both")
        engine_latest = None

    print("server ready!")


@app.on_event("shutdown")
async def shutdown():
    global engine_best, engine_latest
    if engine_best:
        engine_best.cleanup()
    if engine_latest:
        engine_latest.cleanup()


# -- static files + pages --

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get("/api/model-info")
async def model_info():
    return JSONResponse(engine_best.get_model_info())


@app.get("/api/checkpoint-info")
async def checkpoint_info():
    """Returns info about available checkpoints for the UI dropdown."""
    info = {
        'best': {
            'available': engine_best is not None,
            'label': 'Best (lowest loss)',
        },
        'latest': {
            'available': engine_latest is not None,
            'label': 'Latest (most recent)',
        },
    }
    return JSONResponse(info)


@app.get("/api/connections")
async def connections_info():
    """Returns sparse weights for the top 2 connections to render the network diagram."""
    if engine_best:
        return JSONResponse(engine_best.get_sparse_weights(top_k=2))
    return JSONResponse({})


@app.get("/api/audio")
async def serve_audio():
    """
    Serve the extracted audio file for sync playback.
    Returns 404 if audio hasn't been extracted yet.
    """
    if os.path.exists(AUDIO_PATH):
        return FileResponse(AUDIO_PATH, media_type="audio/mpeg")
    return JSONResponse({"error": "audio not extracted yet"}, status_code=404)


@app.get("/api/video")
async def serve_video():
    """
    Serve the original mp4 video for the compare mode.
    """
    from fastapi import Request
    from starlette.responses import StreamingResponse
    import mimetypes

    if not os.path.exists(VIDEO_PATH):
        return JSONResponse({"error": "video not found"}, status_code=404)

    return FileResponse(VIDEO_PATH, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


# -- the main event: websocket streaming --

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    """
    Main streaming endpoint. One connection per browser tab.
    Supports modes: 'best', 'latest', 'compare'
    """
    await ws.accept()
    print("client connected")

    # playback state
    playing = False
    start_time = 0
    start_frame = 0
    current_frame = 0
    fps = DEFAULT_FPS
    mode = 'best'  # 'best', 'latest', or 'compare'

    try:
        while True:
            # check for control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                cmd = json.loads(msg)

                if cmd['type'] == 'play':
                    playing = True
                    start_time = time.time()
                    start_frame = current_frame
                elif cmd['type'] == 'pause':
                    playing = False
                elif cmd['type'] == 'seek':
                    current_frame = int(cmd.get('frame_index', 0))
                    if playing:
                        start_time = time.time()
                        start_frame = current_frame
                    # always send the sought frame immediately
                    await send_frame(ws, current_frame, mode)
                elif cmd['type'] == 'set_fps':
                    fps = max(1, min(60, float(cmd.get('fps', DEFAULT_FPS))))
                    print(f"fps set to {fps}")
                elif cmd['type'] == 'set_mode':
                    mode = cmd.get('mode', 'best')
                    print(f"mode set to {mode}")
                    # send current frame in new mode immediately
                    await send_frame(ws, current_frame, mode)

            except asyncio.TimeoutError:
                pass  # no message, that's fine

            if playing:
                # calculate exact frame based on real-world elapsed time
                elapsed = time.time() - start_time
                target_frame = start_frame + int(elapsed * fps)

                # loop back to start when reach the end + 2 seconds of black screen
                max_frame = engine_best.total_frames + int(fps * 2.0)
                if target_frame >= max_frame:
                    target_frame = 0
                    start_time = time.time()
                    start_frame = 0

                # only generate if we've actually advanced a frame
                if target_frame >= current_frame:
                    current_frame = target_frame
                    await send_frame(ws, current_frame, mode)
                    
                # yield to asyncio
                await asyncio.sleep(0.001)
            else:
                # not playing - just chill and wait for commands.
                
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("client disconnected")
    except Exception as e:
        print(f"websocket error: {e}")


def _encode_result(engine, frame_index):
    """Generate a frame from an engine and encode it for the wire."""
    result = engine.generate_frame(frame_index)
    
    # encode directly to JPEG
    # JPEG at 100% quality is virtually lossless and eliminates the smudging
    # while being 10x faster to encode than WebP for real-time streaming.
    success, buffer = cv2.imencode('.jpg', result['pixels'], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not success:
        raise RuntimeError("failed to encode frame to jpeg")
        
    pixel_b64 = base64.b64encode(buffer).decode('ascii')
    return {
        'frame_index': result['frame_index'],
        'total_frames': result['total_frames'],
        'width': engine.width,
        'height': engine.height,
        'pixels': pixel_b64,
        'activations': result['activations'],
        'inference_ms': result['inference_ms'],
    }


async def send_frame(ws: WebSocket, frame_index: int, mode: str = 'best'):
    """
    Generate one frame and send it over the websocket.

    In 'best' or 'latest' mode, sends a single frame.
    In 'compare' mode, sends both side by side.
    """
    if mode == 'compare':
        best_data = _encode_result(engine_best, frame_index)
        # fall back to best if latest doesn't exist yet
        latest_engine = engine_latest if engine_latest else engine_best
        latest_data = _encode_result(latest_engine, frame_index)

        msg = {
            'type': 'compare_frame',
            'best': best_data,
            'latest': latest_data,
        }
    else:
        # single model mode
        engine = engine_latest if (mode == 'latest' and engine_latest) else engine_best
        data = _encode_result(engine, frame_index)
        msg = {
            'type': 'frame',
            **data,
        }

    await ws.send_text(json.dumps(msg))


if __name__ == '__main__':
    # extract audio from video if it doesn't exist yet
    if not os.path.exists(AUDIO_PATH):
        print("extracting audio from video...")
        try:
            import subprocess
            video_path = os.path.join(os.path.dirname(__file__), '..', 'bad_apple_vid', 'vid.mp4')
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-q:a', '0', '-map', 'a',
                AUDIO_PATH
            ], check=True, capture_output=True)
            print(f"audio saved to {AUDIO_PATH}")
        except Exception as e:
            print(f"couldn't extract audio (ffmpeg missing?): {e}")
            print("the app will work without audio, just no sound")

    uvicorn.run(app, host=HOST, port=PORT)
