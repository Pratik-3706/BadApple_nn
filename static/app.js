/*
 * BadApple_X_NN — frontend
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

// fps tracking
let frameTimestamps = [];
const FPS_WINDOW = 30;

// canvas
const canvas = document.getElementById('video-canvas');
const ctx = canvas.getContext('2d');

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
        document.getElementById('param-display').textContent =
            formatNumber(modelInfo.total_params) + ' params';
        buildNetworkDiagram(modelInfo);
    } catch (e) {
        console.error('model info fetch failed:', e);
    }

    audioPlayer.addEventListener('canplaythrough', () => {
        audioReady = true;
    });
    audioPlayer.addEventListener('error', () => {
        audioReady = false;
    });

    setupControls();
    connectWebSocket();
});


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
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'frame') handleFrame(msg);
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
// frame handler — video + diagram update from the same message
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
    const raw = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }

    const img = ctx.createImageData(width, height);
    for (let i = 0; i < bytes.length; i++) {
        const v = bytes[i];
        const j = i * 4;
        img.data[j] = v;
        img.data[j + 1] = v;
        img.data[j + 2] = v;
        img.data[j + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
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

    if (frameTimestamps.length >= 2) {
        const dt = frameTimestamps[frameTimestamps.length - 1] - frameTimestamps[0];
        const fps = ((frameTimestamps.length - 1) / dt) * 1000;
        document.getElementById('fps-display').textContent = Math.round(fps);
    }
}


// ============================================================
// network diagram
//
// this draws EVERY neuron — all 256 per hidden layer, no cap.
// each neuron gets its own rect in the heatmap strip.
// colors come from the exact values the forward hooks captured.
// ============================================================

let neuronRects = {};       // { 'layer_0': [<rect>, ...], ... }
let currentActivations = {};

function buildNetworkDiagram(info) {
    const svg = document.getElementById('network-svg');
    const container = document.getElementById('network-container');
    const bounds = container.getBoundingClientRect();
    const W = bounds.width || 600;
    const H = bounds.height || 500;

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';

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
    neuronRects = {};
    const layerCenters = [];

    // each neuron = 1 row in the heatmap strip.
    // with 256 neurons, each row is thin but every one is there.
    const neuronsPerLayer = info.hidden_features; // ALL of them
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

        // background rect for the strip
        appendSVG(svg, 'rect', {
            x: sx - 1, y: stripTop - 1,
            width: stripW + 2, height: stripH + 2,
            rx: 3, ry: 3,
            class: 'layer-strip-bg'
        });

        // one rect per neuron — every single one, no skipping
        for (let ni = 0; ni < neuronsPerLayer; ni++) {
            const rect = appendSVG(svg, 'rect', {
                x: sx,
                y: stripTop + ni * neuronH,
                width: stripW,
                height: Math.max(neuronH - 0.3, 0.5),
                fill: '#0f0f18',
                'data-layer': layerName,
                'data-neuron': ni,
            });

            rect.addEventListener('mouseenter', (e) => showTooltip(e, layerName, ni));
            rect.addEventListener('mouseleave', hideTooltip);
            rect.addEventListener('mousemove', (e) => moveTooltip(e));

            neuronRects[layerName].push(rect);
        }

        // layer label
        const label = appendSVG(svg, 'text', {
            x: cx, y: stripTop + stripH + 14, class: 'layer-label'
        });
        label.textContent = `L${li}`;

        // neuron count label — shows this is exact
        const count = appendSVG(svg, 'text', {
            x: cx, y: stripTop + stripH + 23, class: 'layer-count'
        });
        count.textContent = `${neuronsPerLayer}n`;
    }

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

    // ---- connections ----
    // input -> first hidden
    drawInputConnections(svg, inputPositions, layerCenters[0], stripW, stripTop, stripH);

    // hidden -> hidden
    for (let i = 0; i < layerCenters.length - 1; i++) {
        drawHiddenConnections(svg,
            layerCenters[i], layerCenters[i + 1],
            stripW, stripTop, stripH);
    }

    // last hidden -> output
    drawOutputConnections(svg,
        layerCenters[layerCenters.length - 1], stripW,
        stripTop, stripH, outX, outY);
}


// connection drawing helpers — stylized, not all-to-all
// (drawing 256^2 = 65K lines per layer would be a solid block of color)

function createConnection(svg, x1, y1, x2, y2) {
    // sleek bezier S-curve instead of a messy zigzag
    const dx = Math.abs(x2 - x1) * 0.4;
    const d = `M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`;
    
    appendSVG(svg, 'path', {
        class: 'connection-line',
        fill: 'none',
        d: d
    });
}

function drawInputConnections(svg, inputs, toX, stripW, stripTop, stripH) {
    const n = 6;
    const step = stripH / (n + 1);
    for (const pt of inputs) {
        for (let i = 0; i < n; i++) {
            createConnection(svg, pt.x + 9, pt.y, toX - stripW / 2, stripTop + step * (i + 1));
        }
    }
}

function drawHiddenConnections(svg, fromX, toX, stripW, stripTop, stripH) {
    const n = 10;
    const step = stripH / (n + 1);
    for (let i = 0; i < n; i++) {
        const y1 = stripTop + step * (i + 1);
        for (let j = 0; j < 3; j++) {
            const y2 = stripTop + step * (((i + j * 3 + 1) % n) + 1);
            createConnection(svg, fromX + stripW / 2, y1, toX - stripW / 2, y2);
        }
    }
}

function drawOutputConnections(svg, fromX, stripW, stripTop, stripH, toX, toY) {
    const n = 6;
    const step = stripH / (n + 1);
    for (let i = 0; i < n; i++) {
        createConnection(svg, fromX + stripW / 2, stripTop + step * (i + 1), toX - 9, toY);
    }
}


// ============================================================
// activation update — runs every frame, updates ALL neurons
//
// the values come straight from register_forward_hook.
// per-neuron mean activation across all spatial positions.
// this is the exact mathematical quantity — not an approximation.
// ============================================================

function updateActivations(activations) {
    currentActivations = activations;

    for (const [layerName, values] of Object.entries(activations)) {
        const rects = neuronRects[layerName];
        if (!rects) continue;

        // update every single neuron rect — all 256
        const len = Math.min(rects.length, values.length);
        for (let i = 0; i < len; i++) {
            rects[i].setAttribute('fill', valueToColor(values[i]));
        }
    }
}


// ============================================================
// colormap — diverging blue/red
//
// maps the exact activation value to a color.
// sin() layer outputs are bounded to [-1, 1].
// blue = negative, dark = near zero, red = positive.
// ============================================================

function valueToColor(v) {
    // clamp to [-1, 1] — sin() can't exceed this anyway
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
// tooltip — shows the exact value when you hover a neuron
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

        if (audioReady) {
            if (isPlaying) {
                syncAudio(parseInt(seekBar.value));
                audioPlayer.play().catch(() => {});
            } else {
                audioPlayer.pause();
            }
        }
    });

    // seek
    seekBar._dragging = false;
    seekBar.addEventListener('mousedown', () => seekBar._dragging = true);
    seekBar.addEventListener('mouseup', () => {
        seekBar._dragging = false;
        const f = parseInt(seekBar.value);
        send({ type: 'seek', frame_index: f });
        if (audioReady) syncAudio(f);
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
    if (!audioReady || !totalFrames) return;
    const dur = audioPlayer.duration || 219;
    audioPlayer.currentTime = (frameIdx / totalFrames) * dur;
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
