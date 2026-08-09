/*
 * BadApple_X_NN - frontend
 *
 * connects to the websocket, draws frames on canvas,
 * builds the network diagram with EVERY neuron (no subsampling),
 * and updates colors from the EXACT activation values
 * captured by the forward hooks. nothing is estimated.
 */

// ============================================================
// globals
// ============================================================

let ws = null;
let isPlaying = false;
let modelInfo = null;
let proofMode = false;
let currentFps = 29.9;
let currentMode = 'best'; // 'best', 'latest', or 'compare'

// fps tracking
let frameTimestamps = [];
const FPS_WINDOW = 30;

// canvas
const canvas = document.getElementById('video-canvas');
const ctx = canvas.getContext('2d');

// compare canvas
const canvasCompare = document.getElementById('video-canvas-compare');
const ctxCompare = canvasCompare.getContext('2d');

// audio
const audioPlayer = document.getElementById('audio-player');
let audioReady = false;
let totalFrames = 0;


// ============================================================
// boot
// ============================================================

window.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch('/api/model-info');
        modelInfo = await resp.json();
        
        const connResp = await fetch('/api/connections');
        const connections = await connResp.json();
        
        document.getElementById('param-display').textContent =
            formatNumber(modelInfo.total_params) + ' params';
        buildNetworkDiagram(modelInfo, connections);
    } catch (e) {
        console.error('model/connections info fetch failed:', e);
    }

    audioPlayer.addEventListener('canplay', () => {
        audioReady = true;
    });
    audioPlayer.addEventListener('loadedmetadata', () => {
        audioReady = true;
    });
    audioPlayer.addEventListener('error', (e) => {
        console.error("Audio element error:", e);
    });

    setupControls();
    setupModeSelector();
    connectWebSocket();
});


// ============================================================
// mode selector - best / latest / compare
// ============================================================

function setupModeSelector() {
    const select = document.getElementById('model-mode-select');
    select.addEventListener('change', () => {
        currentMode = select.value;
        const serverMode = currentMode === 'compare_real_best' ? 'best' : 
                           currentMode === 'compare_real_latest' ? 'latest' : 
                           currentMode;
        send({ type: 'set_mode', mode: serverMode });
        updateCompareLayout();
    });
}

function updateCompareLayout() {
    const compareFrame = document.getElementById('compare-frame');
    const realFrame = document.getElementById('real-frame');
    const compareLabels = document.getElementById('compare-labels');
    const videoStage = document.getElementById('video-stage');
    const modelTag = document.getElementById('model-tag-main');
    
    const labelBest = document.querySelector('.compare-label--best');
    const labelLatest = document.querySelector('.compare-label--latest');
    const labelReal = document.querySelector('.compare-label--real');

    compareFrame.style.display = 'none';
    realFrame.style.display = 'none';
    compareLabels.style.display = 'none';
    videoStage.classList.remove('video-stage--compare');
    modelTag.style.display = '';
    modelTag.textContent = currentMode.toUpperCase().replace('_', ' ');
    
    labelBest.style.display = 'none';
    labelLatest.style.display = 'none';
    labelReal.style.display = 'none';

    if (currentMode === 'compare') {
        compareFrame.style.display = '';
        compareLabels.style.display = '';
        videoStage.classList.add('video-stage--compare');
        modelTag.style.display = 'none';
        
        labelBest.style.display = '';
        labelLatest.style.display = '';
        
    } else if (currentMode.startsWith('compare_real')) {
        realFrame.style.display = '';
        compareLabels.style.display = '';
        videoStage.classList.add('video-stage--compare');
        modelTag.style.display = 'none';
        
        if (currentMode === 'compare_real_best') {
            labelBest.style.display = '';
        } else {
            labelLatest.style.display = '';
        }
        labelReal.style.display = '';
    }
}


// ============================================================
// websocket
// ============================================================

function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/stream`;

    setStatus('connecting');
    ws = new WebSocket(url);

    ws.onopen = () => {
        setStatus('connected');
        // tell the server what mode we're in
        send({ type: 'set_mode', mode: currentMode });
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'frame') handleFrame(msg);
        else if (msg.type === 'compare_frame') handleCompareFrame(msg);
    };

    ws.onclose = () => {
        setStatus('disconnected');
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => ws.close();
}

function send(cmd) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(cmd));
    }
}


// ============================================================
// frame handler - video + diagram update from the same message
// ============================================================

function handleFrame(msg) {
    const now = performance.now();
    frameTimestamps.push(now);
    while (frameTimestamps.length > FPS_WINDOW) frameTimestamps.shift();

    totalFrames = msg.total_frames;

    // draw pixels
    drawFrame(msg.pixels, msg.width, msg.height);

    // update network with exact activation values
    if (modelInfo && msg.activations) {
        updateActivations(msg.activations);
    }

    // update UI
    updateStats(msg);
}

function drawFrame(b64, width, height) {
    drawFrameOn(canvas, ctx, b64, width, height);
}

function drawFrameOn(targetCanvas, targetCtx, b64, width, height) {
    if (targetCanvas.width !== width || targetCanvas.height !== height) {
        targetCanvas.width = width;
        targetCanvas.height = height;
    }

    const img = new Image();
    img.onload = () => {
        targetCtx.drawImage(img, 0, 0, width, height);
    };
    img.src = 'data:image/jpeg;base64,' + b64;
}

function handleCompareFrame(msg) {
    const now = performance.now();
    frameTimestamps.push(now);
    while (frameTimestamps.length > FPS_WINDOW) frameTimestamps.shift();

    totalFrames = msg.best.total_frames;

    // draw best on left canvas
    drawFrameOn(canvas, ctx, msg.best.pixels, msg.best.width, msg.best.height);

    // draw latest on right canvas
    drawFrameOn(canvasCompare, ctxCompare, msg.latest.pixels, msg.latest.width, msg.latest.height);

    // update network activations from the best model (it's the reference)
    if (modelInfo && msg.best.activations) {
        updateActivations(msg.best.activations);
    }

    // use the best model's stats for display
    updateStats(msg.best);
}

function updateStats(msg) {
    document.getElementById('frame-counter').textContent =
        `frame ${msg.frame_index} / ${msg.total_frames}`;

    const seekBar = document.getElementById('seek-bar');
    if (!seekBar._dragging) {
        seekBar.max = msg.total_frames - 1;
        seekBar.value = msg.frame_index;
        // update progress bar width
        const pct = msg.frame_index / Math.max(msg.total_frames - 1, 1) * 100;
        document.getElementById('seek-progress').style.width = pct + '%';
    }

    document.getElementById('inference-display').textContent =
        msg.inference_ms.toFixed(1) + ' ms';
        
    // soft-sync real video to the exact frame the NN is currently rendering
    // this prevents the HTML5 video from drifting ahead or behind the websocket stream
    const realVideo = document.getElementById('real-video');
    if (realVideo && currentMode.startsWith('compare_real')) {
        if (msg.frame_index >= msg.total_frames - 1) {
            // hide real video at the end to match the NN rendering black
            realVideo.style.opacity = '0';
        } else {
            realVideo.style.opacity = '1';
            if (!realVideo.paused) {
                const dur = realVideo.duration || 219.0;
                const targetTime = (msg.frame_index / msg.total_frames) * dur;
                if (Math.abs(realVideo.currentTime - targetTime) > 0.1) {
                    realVideo.currentTime = targetTime;
                }
            }
        }
    }

    if (frameTimestamps.length >= 2) {
        const dt = frameTimestamps[frameTimestamps.length - 1] - frameTimestamps[0];
        const fps = ((frameTimestamps.length - 1) / dt) * 1000;
        document.getElementById('fps-display').textContent = Math.round(fps);
    }
}


// ============================================================
// network diagram
//
// this draws EVERY neuron - all 512 per hidden layer, no cap.
// each neuron gets its own rect in the heatmap strip.
// colors come from the exact values the forward hooks captured.
// ============================================================

let neuronRects = {};       // { 'layer_0': [<rect>, ...], ... }
let currentActivations = {};

function buildNetworkDiagram(info, connections = {}) {
    const svg = document.getElementById('network-svg');
    const container = document.getElementById('network-container');
    const bounds = container.getBoundingClientRect();
    const W = bounds.width || 600;
    const H = bounds.height || 500;

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';

    // setup canvas context
    const netCanvas = document.getElementById('network-canvas');
    netCanvas.width = W;
    netCanvas.height = H;
    const netCtx = netCanvas.getContext('2d', { alpha: true });
    
    // clear it out
    netCtx.clearRect(0, 0, W, H);
    
    const numHidden = info.layer_names.length;
    const totalCols = numHidden + 2; // input + hidden layers + output
    const colW = W / totalCols;
    const padY = 36;
    const usableH = H - padY * 2;

    // ---- input nodes (t, x, y) ----
    const inX = colW * 0.5;
    const inSpacing = usableH / (info.input_dim + 1);
    const inputPositions = [];

    for (let i = 0; i < info.input_dim; i++) {
        const cy = padY + inSpacing * (i + 1);
        inputPositions.push({ x: inX, y: cy });

        appendSVG(svg, 'circle', {
            cx: inX, cy, r: 9, class: 'node-io'
        });
        const lbl = appendSVG(svg, 'text', {
            x: inX, y: cy, class: 'node-io-label'
        });
        lbl.textContent = info.input_labels[i];
    }

    // ---- hidden layers: ALL neurons, no subsampling ----
    neuronRects = {}; // { 'layer_0': [{x, y, w, h}, ...], ... }
    const layerCenters = [];

    // each neuron = 1 row in the heatmap strip.
    const neuronsPerLayer = info.hidden_features; 
    const neuronH = Math.max(usableH / neuronsPerLayer, 0.5);
    const stripH = neuronH * neuronsPerLayer;
    const stripTop = padY + (usableH - stripH) / 2;
    const stripW = colW * 0.5;

    for (let li = 0; li < numHidden; li++) {
        const cx = colW * (li + 1.5);
        layerCenters.push(cx);

        const layerName = `layer_${li}`;
        neuronRects[layerName] = [];
        const sx = cx - stripW / 2;

        // background rect for the strip (can draw on canvas now)
        netCtx.fillStyle = '#1e1e24';
        netCtx.beginPath();
        netCtx.roundRect(sx - 1, stripTop - 1, stripW + 2, stripH + 2, 3);
        netCtx.fill();

        // store coords for every single neuron, no skipping
        for (let ni = 0; ni < neuronsPerLayer; ni++) {
            neuronRects[layerName].push({
                x: sx,
                y: stripTop + ni * neuronH,
                w: stripW,
                h: Math.max(neuronH - 0.3, 0.5)
            });
        }

        // layer label (keep in SVG for crisp text)
        const label = appendSVG(svg, 'text', {
            x: cx, y: stripTop + stripH + 14, class: 'layer-label'
        });
        label.textContent = `L${li}`;

        // neuron count label 
        const count = appendSVG(svg, 'text', {
            x: cx, y: stripTop + stripH + 23, class: 'layer-count'
        });
        count.textContent = `${neuronsPerLayer}n`;
    }

    // setup canvas mouse tracking for tooltips
    netCanvas.style.pointerEvents = 'auto'; // allow mouse events on canvas
    netCanvas.addEventListener('mousemove', (e) => {
        const rect = netCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // find which neuron we are hovering
        let found = false;
        for (const [layerName, rects] of Object.entries(neuronRects)) {
            // quick check bounding box of whole layer first
            const first = rects[0];
            const last = rects[rects.length - 1];
            if (mouseX >= first.x && mouseX <= first.x + first.w &&
                mouseY >= first.y && mouseY <= last.y + last.h) {
                
                // inside this layer, find exact neuron
                for (let i = 0; i < rects.length; i++) {
                    const r = rects[i];
                    if (mouseY >= r.y && mouseY <= r.y + r.h) {
                        showTooltip(e, layerName, i);
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
        }

        if (!found) {
            hideTooltip();
        } else {
            moveTooltip(e);
        }
    });
    netCanvas.addEventListener('mouseleave', hideTooltip);

    // ---- output node ----
    const outX = colW * (totalCols - 0.5);
    const outY = padY + usableH / 2;

    appendSVG(svg, 'circle', {
        cx: outX, cy: outY, r: 9, class: 'node-io'
    });
    const outLbl = appendSVG(svg, 'text', {
        x: outX, y: outY, class: 'node-io-label'
    });
    outLbl.textContent = 'px';

    // ---- real connections ----
    const connCanvas = document.getElementById('connections-canvas');
    if (connCanvas && Object.keys(connections).length > 0) {
        connCanvas.width = W;
        connCanvas.height = H;
        const connCtx = connCanvas.getContext('2d');
        connCtx.clearRect(0, 0, W, H);
        
        // draw the sparse real connections
        for (let li = 0; li < numHidden; li++) {
            const layerName = `layer_${li}`;
            const layerConns = connections[layerName];
            if (!layerConns) continue;
            
            const isFirst = (li === 0);
            
            // X coordinates
            const fromX = isFirst ? inX + 9 : layerCenters[li - 1] + stripW / 2;
            const toX = layerCenters[li] - stripW / 2;
            
            // source Y coords (inputs or previous layer)
            const getSourceY = (srcIdx) => {
                if (isFirst) {
                    return padY + inSpacing * (srcIdx + 1);
                }
                const prevName = `layer_${li - 1}`;
                const prevRects = neuronRects[prevName];
                if (prevRects && prevRects[srcIdx]) {
                    return prevRects[srcIdx].y + prevRects[srcIdx].h / 2;
                }
                return padY + usableH / 2;
            };
            
            const destRects = neuronRects[layerName];
            if (!destRects) continue;
            
            // draw lines
            for (let destIdx = 0; destIdx < layerConns.length; destIdx++) {
                const conns = layerConns[destIdx];
                const destY = destRects[destIdx].y + destRects[destIdx].h / 2;
                
                for (const [srcIdx, weight] of conns) {
                    const srcY = getSourceY(srcIdx);
                    
                    connCtx.beginPath();
                    connCtx.moveTo(fromX, srcY);
                    connCtx.lineTo(toX, destY);
                    
                    // color by weight (red/blue brutalist)
                    const normalizedW = Math.max(-1, Math.min(1, weight * 10)); // arbitrarily scale up for color visibility
                    if (normalizedW >= 0) {
                        connCtx.strokeStyle = `rgba(220, 38, 38, ${0.1 + normalizedW * 0.4})`;
                    } else {
                        connCtx.strokeStyle = `rgba(30, 64, 175, ${0.1 + (-normalizedW) * 0.4})`;
                    }
                    
                    connCtx.lineWidth = 1.0;
                    connCtx.stroke();
                }
            }
        }
        
        // Output layer connections (from last hidden layer to output)
        const lastLayerName = `layer_${numHidden}`;
        const outputConns = connections[lastLayerName];
        if (outputConns && outputConns.length > 0) {
            const fromX = layerCenters[numHidden - 1] + stripW / 2;
            const prevRects = neuronRects[`layer_${numHidden - 1}`];
            
            for (const [srcIdx, weight] of outputConns[0]) {
                if (prevRects && prevRects[srcIdx]) {
                    const srcY = prevRects[srcIdx].y + prevRects[srcIdx].h / 2;
                    connCtx.beginPath();
                    connCtx.moveTo(fromX, srcY);
                    connCtx.lineTo(outX - 9, outY);
                    
                    const normalizedW = Math.max(-1, Math.min(1, weight * 10));
                    if (normalizedW >= 0) {
                        connCtx.strokeStyle = `rgba(220, 38, 38, ${0.2 + normalizedW * 0.6})`;
                    } else {
                        connCtx.strokeStyle = `rgba(30, 64, 175, ${0.2 + (-normalizedW) * 0.6})`;
                    }
                    connCtx.lineWidth = 1.0;
                    connCtx.stroke();
                }
            }
        }
    }
}


// ============================================================
// activation update - runs every frame, updates ALL neurons
//
// the values come straight from register_forward_hook.
// per-neuron mean activation across all spatial positions.
// this is the exact mathematical quantity - not an approximation.
// ============================================================

function updateActivations(activations) {
    currentActivations = activations;

    const netCanvas = document.getElementById('network-canvas');
    if (!netCanvas) return;
    const netCtx = netCanvas.getContext('2d', { alpha: true });

    for (const [layerName, values] of Object.entries(activations)) {
        const rects = neuronRects[layerName];
        if (!rects) continue;

        const len = Math.min(rects.length, values.length);
        for (let i = 0; i < len; i++) {
            const r = rects[i];
            netCtx.fillStyle = valueToColor(values[i]);
            netCtx.fillRect(r.x, r.y, r.w, r.h);
        }
    }
}


// ============================================================
// colormap - diverging blue/red
//
// maps the exact activation value to a color.
// sin() layer outputs are bounded to [-1, 1].
// blue = negative, dark = near zero, red = positive.
// ============================================================

function valueToColor(v) {
    // clamp to [-1, 1] - sin() can't exceed this anyway
    v = Math.max(-1, Math.min(1, v));

    // we use a perceptually uniform-ish diverging map:
    // negative: interpolate from dark (#0f0f18) to blue (#1e40af)
    // positive: interpolate from dark (#0f0f18) to red (#dc2626)

    let r, g, b;

    if (v >= 0) {
        const t = v;
        // dark(15,15,24) -> red(220,38,38)
        r = Math.round(15 + 205 * t);
        g = Math.round(15 + 23 * t);
        b = Math.round(24 + 14 * t);
    } else {
        const t = -v;
        // dark(15,15,24) -> blue(30,64,175)
        r = Math.round(15 + 15 * t);
        g = Math.round(15 + 49 * t);
        b = Math.round(24 + 151 * t);
    }

    return `rgb(${r},${g},${b})`;
}


// ============================================================
// tooltip - shows the exact value when you hover a neuron
// ============================================================

function showTooltip(e, layerName, neuronIdx) {
    const tip = document.getElementById('neuron-tooltip');
    const vals = currentActivations[layerName];
    if (!vals) return;

    document.getElementById('tooltip-layer').textContent = layerName;
    document.getElementById('tooltip-neuron').textContent = `#${neuronIdx}`;

    const v = vals[neuronIdx];
    document.getElementById('tooltip-value').textContent =
        v !== undefined ? v.toFixed(8) : 'n/a';

    tip.style.display = 'block';
    positionTooltip(e);
}

function moveTooltip(e) {
    positionTooltip(e);
}

function hideTooltip() {
    document.getElementById('neuron-tooltip').style.display = 'none';
}

function positionTooltip(e) {
    const tip = document.getElementById('neuron-tooltip');
    const pad = 14;
    let x = e.clientX + pad;
    let y = e.clientY - 10;

    // keep on screen
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    if (x + tw > window.innerWidth - 10) x = e.clientX - tw - pad;
    if (y + th > window.innerHeight - 10) y = window.innerHeight - th - 10;

    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
}


// ============================================================
// controls
// ============================================================

function setupControls() {
    const playBtn = document.getElementById('play-btn');
    const playIcon = document.getElementById('play-icon');
    const pauseIcon = document.getElementById('pause-icon');
    const seekBar = document.getElementById('seek-bar');
    const fpsSlider = document.getElementById('fps-slider');
    const fpsValue = document.getElementById('fps-value');
    const proofToggle = document.getElementById('proof-mode-toggle');

    // play/pause
    playBtn.addEventListener('click', () => {
        isPlaying = !isPlaying;
        playIcon.style.display = isPlaying ? 'none' : 'block';
        pauseIcon.style.display = isPlaying ? 'block' : 'none';
        playBtn.classList.toggle('playing', isPlaying);
        send({ type: isPlaying ? 'play' : 'pause' });

        if (isPlaying) {
            syncAudio(parseInt(seekBar.value));
            audioPlayer.play().catch(e => console.error("Audio play error:", e));
            const realVideo = document.getElementById('real-video');
            if (realVideo && currentMode.startsWith('compare_real')) realVideo.play().catch(()=>{});
        } else {
            audioPlayer.pause();
            const realVideo = document.getElementById('real-video');
            if (realVideo) realVideo.pause();
        }
    });

    // seek
    seekBar._dragging = false;
    seekBar.addEventListener('mousedown', () => seekBar._dragging = true);
    seekBar.addEventListener('mouseup', () => {
        seekBar._dragging = false;
        const f = parseInt(seekBar.value);
        send({ type: 'seek', frame_index: f });
        if (isPlaying) syncAudio(f);
    });
    seekBar.addEventListener('input', () => {
        if (seekBar._dragging) {
            send({ type: 'seek', frame_index: parseInt(seekBar.value) });
            const pct = seekBar.value / Math.max(seekBar.max, 1) * 100;
            document.getElementById('seek-progress').style.width = pct + '%';
        }
    });

    // fps
    fpsSlider.addEventListener('input', () => {
        currentFps = parseInt(fpsSlider.value);
        fpsValue.textContent = currentFps;
        send({ type: 'set_fps', fps: currentFps });
    });

    // proof mode
    proofToggle.addEventListener('change', () => {
        proofMode = proofToggle.checked;
    });

    // keyboard
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') { e.preventDefault(); playBtn.click(); }
        else if (e.code === 'ArrowRight') {
            send({ type: 'seek', frame_index: parseInt(seekBar.value) + 10 });
        }
        else if (e.code === 'ArrowLeft') {
            send({ type: 'seek', frame_index: Math.max(0, parseInt(seekBar.value) - 10) });
        }
    });
}

function syncAudio(frameIdx) {
    if (!totalFrames) return;
    const dur = audioPlayer.duration && !isNaN(audioPlayer.duration) ? audioPlayer.duration : 219.0;
    const t = (frameIdx / totalFrames) * dur;
    audioPlayer.currentTime = t;
    
    const realVideo = document.getElementById('real-video');
    if (realVideo && currentMode.startsWith('compare_real')) {
        realVideo.currentTime = t;
    }
}


// ============================================================
// status
// ============================================================

function setStatus(state) {
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    dot.className = 'status-dot ' + state;
    txt.textContent = state;
}


// ============================================================
// helpers
// ============================================================

function appendSVG(parent, tag, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    parent.appendChild(el);
    return el;
}

function formatNumber(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
    return String(n);
}

// rebuild diagram on resize
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (modelInfo) buildNetworkDiagram(modelInfo);
    }, 300);
});
