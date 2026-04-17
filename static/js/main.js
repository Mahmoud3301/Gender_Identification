/* ═══════════════════════════════════════════════════════════
   VoiceID v2 — CNN vs GMM | AI402 Spring 2026
   Features: Upload + Live Recording
   ═══════════════════════════════════════════════════════════ */
'use strict';

// ─── DOM ───
const audioInput      = document.getElementById('audioInput');
const uploadZone      = document.getElementById('uploadZone');
const uploadFilename  = document.getElementById('uploadFilename');
const analyzeBtn      = document.getElementById('analyzeBtn');
const btnLoader       = document.getElementById('btnLoader');
const analyzeRecBtn   = document.getElementById('analyzeRecBtn');
const btnLoader2      = document.getElementById('btnLoader2');
const resultPanel     = document.getElementById('resultPanel');
const errorPanel      = document.getElementById('errorPanel');
const errorMsg        = document.getElementById('errorMsg');
const historyList     = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistory');

const modelTabs  = document.querySelectorAll('.model-tab');
const modeTabs   = document.querySelectorAll('.mode-tab');
const uploadSection = document.getElementById('uploadSection');
const recordSection = document.getElementById('recordSection');

// Record elements
const recStartBtn   = document.getElementById('recStartBtn');
const recStopBtn    = document.getElementById('recStopBtn');
const recPlayBtn    = document.getElementById('recPlayBtn');
const recStatus     = document.getElementById('recStatus');
const recTimer      = document.getElementById('recTimer');
const recCanvas     = document.getElementById('recCanvas');
const recAudioPlayer= document.getElementById('recAudioPlayer');

// ─── State ───
let selectedModel = 'cnn';
let selectedFile  = null;
let predHistory   = [];

// Recording state
let mediaRecorder   = null;
let audioChunks     = [];
let recordedBlob    = null;
let timerInterval   = null;
let recSeconds      = 0;
let animFrameId     = null;
let audioCtx        = null;
let analyserNode    = null;
let micStream       = null;

// ─────────────────────────────────────────────
// Model Tab Switching
// ─────────────────────────────────────────────
modelTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    modelTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    selectedModel = tab.dataset.model;
    // Show correct model card
    document.getElementById('card-cnn').style.display = selectedModel === 'cnn' ? 'block' : 'none';
    document.getElementById('card-gmm').style.display = selectedModel === 'gmm' ? 'block' : 'none';
  });
});

// ─────────────────────────────────────────────
// Input Mode Tabs (Upload / Record)
// ─────────────────────────────────────────────
modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    modeTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const mode = tab.dataset.mode;
    uploadSection.style.display = mode === 'upload' ? 'block' : 'none';
    recordSection.style.display = mode === 'record' ? 'block' : 'none';
    hideResult(); hideError();
  });
});

// ─────────────────────────────────────────────
// Upload Logic
// ─────────────────────────────────────────────
uploadZone.addEventListener('click', () => audioInput.click());
audioInput.addEventListener('change', () => {
  if (audioInput.files.length > 0) handleFile(audioInput.files[0]);
});
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  const ok = /\.(wav|mp3|ogg|flac|m4a)$/i.test(file.name);
  if (!ok) { showError('Unsupported format. Use WAV, MP3, OGG, FLAC, or M4A.'); return; }
  selectedFile = file;
  uploadZone.classList.add('has-file');
  uploadFilename.textContent = `✓ ${file.name}  (${formatBytes(file.size)})`;
  uploadFilename.style.display = 'block';
  document.querySelectorAll('.upload-waveform span').forEach(s => s.style.background = 'var(--accent-green)');
  analyzeBtn.disabled = false;
  hideResult(); hideError();
}

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  setLoading('upload', true);
  hideResult(); hideError();
  const fd = new FormData();
  fd.append('audio', selectedFile);
  fd.append('model', selectedModel);
  try {
    const res  = await fetch('/predict', { method:'POST', body:fd });
    const data = await res.json();
    if (data.error) showError(data.error);
    else { displayResult(data); addToHistory(data, selectedFile.name); }
  } catch { showError('Network error — is Flask running?'); }
  finally { setLoading('upload', false); }
});

// ─────────────────────────────────────────────
// Recording Logic
// ─────────────────────────────────────────────
recStartBtn.addEventListener('click', startRecording);
recStopBtn.addEventListener('click',  stopRecording);
recPlayBtn.addEventListener('click',  playRecording);

async function startRecording() {
  try {
    micStream   = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch(e) {
    showError('Microphone access denied. Please allow microphone access and try again.');
    return;
  }

  audioChunks  = [];
  recordedBlob = null;
  recSeconds   = 0;

  // Audio visualizer
  audioCtx     = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 256;
  const src = audioCtx.createMediaStreamSource(micStream);
  src.connect(analyserNode);
  drawRecWaveform();

  // Recorder
  mediaRecorder = new MediaRecorder(micStream);
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start(100);

  // UI
  recStartBtn.disabled = true;
  recStopBtn.disabled  = false;
  recPlayBtn.disabled  = true;
  analyzeRecBtn.disabled = true;
  recStartBtn.classList.add('recording');
  document.getElementById('recordZone').classList.add('recording');
  recStatus.textContent = '● RECORDING...';
  recStatus.className = 'rec-status active';

  // Timer
  timerInterval = setInterval(() => {
    recSeconds++;
    const m = Math.floor(recSeconds/60).toString().padStart(2,'0');
    const s = (recSeconds%60).toString().padStart(2,'0');
    recTimer.textContent = `${m}:${s}`;
    // Auto-stop at 30s
    if (recSeconds >= 30) stopRecording();
  }, 1000);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  micStream?.getTracks().forEach(t => t.stop());
  clearInterval(timerInterval);
  cancelAnimationFrame(animFrameId);
  audioCtx?.close();

  recStartBtn.disabled = false;
  recStopBtn.disabled  = true;
  recStartBtn.classList.remove('recording');
  document.getElementById('recordZone').classList.remove('recording');
  recStatus.textContent = 'Processing recording...';
  recStatus.className = 'rec-status';
}

function onRecordingStop() {
  recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
  const url    = URL.createObjectURL(recordedBlob);
  recAudioPlayer.src = url;
  recAudioPlayer.style.display = 'block';

  recPlayBtn.disabled    = false;
  analyzeRecBtn.disabled = false;
  recStatus.textContent  = `✓ Recording complete (${recSeconds}s) — ready to analyze`;
  recStatus.className    = 'rec-status done';

  // Clear canvas and draw flat line
  const ctx = recCanvas.getContext('2d');
  ctx.clearRect(0, 0, recCanvas.width, recCanvas.height);
  ctx.strokeStyle = 'rgba(0,230,118,0.4)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, recCanvas.height/2);
  ctx.lineTo(recCanvas.width, recCanvas.height/2);
  ctx.stroke();
}

function playRecording() {
  recAudioPlayer.play();
}

function drawRecWaveform() {
  if (!analyserNode) return;
  const ctx    = recCanvas.getContext('2d');
  const buf    = new Uint8Array(analyserNode.frequencyBinCount);
  const W      = recCanvas.width;
  const H      = recCanvas.height;

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(buf);
    ctx.fillStyle = 'rgba(8,12,16,0.85)';
    ctx.fillRect(0, 0, W, H);
    ctx.lineWidth   = 2;
    ctx.strokeStyle = '#f44336';
    ctx.beginPath();
    const sliceW = W / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i] / 128.0;
      const y = (v * H) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.lineTo(W, H/2);
    ctx.stroke();
  }
  draw();
}

analyzeRecBtn.addEventListener('click', async () => {
  if (!recordedBlob) return;
  setLoading('record', true);
  hideResult(); hideError();

  // Convert blob to base64
  const reader = new FileReader();
  reader.onloadend = async () => {
    const b64 = reader.result;
    const fd  = new FormData();
    fd.append('audio_blob', b64);
    fd.append('model', selectedModel);
    try {
      const res  = await fetch('/predict', { method:'POST', body:fd });
      const data = await res.json();
      if (data.error) showError(data.error);
      else { displayResult(data); addToHistory(data, 'live-recording.webm'); }
    } catch { showError('Network error — is Flask running?'); }
    finally { setLoading('record', false); }
  };
  reader.readAsDataURL(recordedBlob);
});

// ─────────────────────────────────────────────
// Display Result
// ─────────────────────────────────────────────
function displayResult(data) {
  const { prediction: gender, confidence: conf, probabilities: probs,
          stats = {}, model: modelName, inference_ms, demo_mode } = data;

  // Avatar
  const avatar     = document.getElementById('resultAvatar');
  const avatarIcon = document.getElementById('avatarIcon');
  const genderEl   = document.getElementById('resultGender');
  avatar.className = `result-avatar ${gender}`;
  avatarIcon.textContent = gender === 'male' ? '♂' : '♀';
  genderEl.textContent   = gender.toUpperCase();
  genderEl.className     = `result-gender ${gender}`;

  // Badge
  document.getElementById('resultModelBadge').textContent =
    modelName.toUpperCase() + (demo_mode ? ' (DEMO)' : '');

  // Confidence
  document.getElementById('confValue').textContent = (conf * 100).toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('confBar').style.width = (conf * 100) + '%';
  }, 50);

  // Probability bars
  const pM = (probs.male   || 0) * 100;
  const pF = (probs.female || 0) * 100;
  setTimeout(() => {
    document.getElementById('probMaleBar').style.width   = pM + '%';
    document.getElementById('probFemaleBar').style.width = pF + '%';
  }, 80);
  document.getElementById('probMalePct').textContent   = pM.toFixed(1) + '%';
  document.getElementById('probFemalePct').textContent = pF.toFixed(1) + '%';

  // Stats grid
  const sg = document.getElementById('statsGrid');
  sg.innerHTML = '';
  const statsDisplay = [
    { title: 'DURATION',         value: (stats.duration || '—') + ' s' },
    { title: 'PITCH (mean)',     value: (stats.pitch_mean || '—') + ' Hz' },
    { title: 'PITCH (median)',   value: (stats.pitch_median || '—') + ' Hz' },
    { title: 'CENTROID',         value: (stats.centroid_mean || '—') + ' Hz' },
    { title: 'BANDWIDTH',        value: (stats.bandwidth_mean || '—') + ' Hz' },
    { title: 'VOICED',           value: stats.voiced_ratio != null ? stats.voiced_ratio + '%' : '—' },
  ];
  statsDisplay.forEach(s => {
    sg.innerHTML += `<div class="stat-card">
      <div class="stat-title">${s.title}</div>
      <div class="stat-value">${s.value}</div>
    </div>`;
  });

  // Inference time
  document.getElementById('inferenceTime').textContent =
    inference_ms != null ? inference_ms + ' ms' : '—';

  resultPanel.style.display = 'block';
}

// ─────────────────────────────────────────────
// History
// ─────────────────────────────────────────────
function addToHistory(data, filename) {
  predHistory.unshift({
    filename: filename.length > 18 ? filename.slice(0,16)+'…' : filename,
    prediction: data.prediction, model: data.model,
    conf: data.confidence
  });
  if (predHistory.length > 25) predHistory.pop();
  renderHistory();
}

function renderHistory() {
  if (!predHistory.length) {
    historyList.innerHTML = '<div class="history-empty">No predictions yet</div>';
    return;
  }
  historyList.innerHTML = predHistory.map(item => `
    <div class="history-item">
      <span class="hi-filename" title="${item.filename}">${item.filename}</span>
      <span class="hi-pred ${item.prediction}">${item.prediction}</span>
      <span class="hi-model">${item.model.toUpperCase()} ${(item.conf*100).toFixed(0)}%</span>
    </div>
  `).join('');
}

clearHistoryBtn.addEventListener('click', () => { predHistory = []; renderHistory(); });

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
function setLoading(source, isLoading) {
  if (source === 'upload') {
    analyzeBtn.disabled = isLoading;
    document.querySelector('#analyzeBtn .btn-text').style.display = isLoading ? 'none' : 'inline';
    btnLoader.style.display  = isLoading ? 'inline-block' : 'none';
  } else {
    analyzeRecBtn.disabled   = isLoading;
    document.querySelector('#analyzeRecBtn .btn-text').style.display = isLoading ? 'none' : 'inline';
    btnLoader2.style.display = isLoading ? 'inline-block' : 'none';
  }
}

function hideResult() { resultPanel.style.display = 'none'; }
function hideError()  { errorPanel.style.display  = 'none'; }
function showError(m) { errorMsg.textContent = m; errorPanel.style.display = 'flex'; }
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(2) + ' MB';
}

// ─── Init: show CNN card ───
document.getElementById('card-cnn').style.display = 'block';
document.getElementById('card-gmm').style.display = 'none';
