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

app = FastAPI(title="BadApple_X_NN")

# the engine gets created at startup, not import time.
# this way the model loads once and stays warm.
engine = None


@app.on_event("startup")
async def startup():
    global engine
    print("loading inference engine...")
    engine = InferenceEngine()
    print("server ready!")


@app.on_event("shutdown")
async def shutdown():
    global engine
    if engine:
        engine.cleanup()


# -- static files + pages --

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get("/api/model-info")
async def model_info():

    return JSONResponse(engine.get_model_info())


@app.get("/api/audio")
async def serve_audio():
    """
    Serve the extracted audio file for sync playback.
    Returns 404 if audio hasn't been extracted yet.
    """
    if os.path.exists(AUDIO_PATH):
        return FileResponse(AUDIO_PATH, media_type="audio/mpeg")
    return JSONResponse({"error": "audio not extracted yet"}, status_code=404)


# -- the main event: websocket streaming --

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    """
    Main streaming endpoint. One connection per browser tab.
    """
    await ws.accept()
    print("client connected")

    # playback state
    playing = False
    start_time = 0
    start_frame = 0
    current_frame = 0
    fps = DEFAULT_FPS

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
                    await send_frame(ws, current_frame)
                elif cmd['type'] == 'set_fps':
                    fps = max(1, min(60, float(cmd.get('fps', DEFAULT_FPS))))
                    print(f"fps set to {fps}")

            except asyncio.TimeoutError:
                pass  # no message, that's fine

            if playing:
                # calculate exact frame based on real-world elapsed time
                elapsed = time.time() - start_time
                target_frame = start_frame + int(elapsed * fps)

                # loop back to start when reach the end + 2 seconds of black screen
                max_frame = engine.total_frames + int(fps * 2.0)
                if target_frame >= max_frame:
                    target_frame = 0
                    start_time = time.time()
                    start_frame = 0

                # only generate if we've actually advanced a frame
                if target_frame >= current_frame:
                    current_frame = target_frame
                    await send_frame(ws, current_frame)
                    
                # yield to asyncio
                await asyncio.sleep(0.001)
            else:
                # not playing — just chill and wait for commands.
                
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("client disconnected")
    except Exception as e:
        print(f"websocket error: {e}")


async def send_frame(ws: WebSocket, frame_index: int):
    """
    Generate one frame and send it over the websocket.

    The pixel data goes as base64-encoded uint8 bytes.
    Activations go as plain JSON arrays — they're small enough
    that JSON overhead doesn't matter at 30fps.
    """
    result = engine.generate_frame(frame_index)

    # encode pixel data as base64
    pixel_bytes = result['pixels'].tobytes()
    pixel_b64 = base64.b64encode(pixel_bytes).decode('ascii')

    msg = {
        'type': 'frame',
        'frame_index': result['frame_index'],
        'total_frames': result['total_frames'],
        'width': engine.width,
        'height': engine.height,
        'pixels': pixel_b64,
        'activations': result['activations'],
        'inference_ms': result['inference_ms'],
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
