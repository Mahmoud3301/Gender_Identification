"""
Gender Identification — Flask Web Application v2
AI402 Spring 2026 | Case Study 08

Models: CNN (Deep Learning) vs GMM (Classical Statistical)
Features: MFCC + Mel-Spectrogram + Spectral Centroid/Bandwidth + Pitch F0

Run:  python app.py
"""

import os, json, tempfile, time, base64
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXT = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ─────────────────────────────────────────────
# Model definition (must match notebook)
# ─────────────────────────────────────────────
class GenderCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
            )
        self.features = nn.Sequential(
            conv_block(1, 32), conv_block(32, 64), conv_block(64, 128)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 64),      nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ─────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Flask] Device: {DEVICE}')

MODELS_LOADED = False
feat_cfg = scaler = le = gmm_male = gmm_female = cnn_model = None

def load_all():
    global feat_cfg, scaler, le, gmm_male, gmm_female, cnn_model, MODELS_LOADED
    try:
        with open(os.path.join(MODELS_DIR, 'feat_config.json')) as f:
            feat_cfg = json.load(f)
        scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        le        = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        gmm_male  = joblib.load(os.path.join(MODELS_DIR, 'gmm_male.pkl'))
        gmm_female= joblib.load(os.path.join(MODELS_DIR, 'gmm_female.pkl'))
        cnn_model = GenderCNN().to(DEVICE)
        cnn_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, 'cnn_best.pth'), map_location=DEVICE))
        cnn_model.eval()
        MODELS_LOADED = True
        print('[Flask] All models loaded ✅')
    except Exception as e:
        print(f'[Flask] DEMO MODE — {e}')

load_all()

# ─────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────
def _get_cfg():
    if feat_cfg:
        return feat_cfg
    return {'sr':22050,'duration':3.0,'n_mfcc':40,'n_mels':128,
            'img_size':[128,128],'fmax':8000,'classes':['female','male']}

def _load_audio(path):
    cfg = _get_cfg()
    sr, dur = cfg['sr'], cfg['duration']
    y, _ = librosa.load(path, sr=sr, duration=dur, mono=True)
    target = int(sr * dur)
    return np.pad(y, (0, max(0, target-len(y))))[:target], sr

def _flat_features(y, sr, n_mfcc=40):
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    dmfcc  = librosa.feature.delta(mfcc)
    d2mfcc = librosa.feature.delta(mfcc, order=2)
    mf  = np.concatenate([mfcc.mean(1),  mfcc.std(1)])
    df_ = np.concatenate([dmfcc.mean(1), dmfcc.std(1)])
    d2f = np.concatenate([d2mfcc.mean(1),d2mfcc.std(1)])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    ro   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    freq_feat = np.array([cent.mean(),cent.std(),bw.mean(),bw.std(),ro.mean(),ro.std()])
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    vf0 = f0[~np.isnan(f0)]
    pitch_feat = np.array([
        np.mean(vf0) if len(vf0)>0 else 0,
        np.std(vf0)  if len(vf0)>0 else 0,
        np.median(vf0) if len(vf0)>0 else 0
    ])
    return np.concatenate([mf, df_, d2f, freq_feat, pitch_feat])

def _mel_spec(y, sr):
    from skimage.transform import resize
    cfg = _get_cfg()
    n_mels, fmax = cfg['n_mels'], cfg['fmax']
    img_size = tuple(cfg['img_size'])
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return resize(mel_db.astype(np.float32), img_size, anti_aliasing=True)

def _audio_stats(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    vf0  = f0[~np.isnan(f0)]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    return {
        'duration':      round(float(len(y)/sr), 2),
        'pitch_mean':    round(float(np.mean(vf0)) if len(vf0)>0 else 0, 1),
        'pitch_median':  round(float(np.median(vf0)) if len(vf0)>0 else 0, 1),
        'centroid_mean': round(float(np.mean(cent)), 1),
        'bandwidth_mean':round(float(np.mean(bw)), 1),
        'voiced_ratio':  round(float(np.sum(~np.isnan(f0))/len(f0))*100, 1),
    }

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict_gender(audio_path, model_type='cnn'):
    t0 = time.time()
    try:
        y, sr = _load_audio(audio_path)
        stats = _audio_stats(y, sr)

        if not MODELS_LOADED:
            import random
            label = random.choice(['male', 'female'])
            conf  = round(random.uniform(0.70, 0.96), 4)
            return {
                'prediction': label, 'confidence': conf,
                'probabilities': {'male': conf if label=='male' else 1-conf,
                                  'female': 1-conf if label=='male' else conf},
                'model': model_type, 'stats': stats,
                'inference_ms': round((time.time()-t0)*1000,1), 'demo_mode': True
            }

        if model_type == 'gmm':
            feat   = _flat_features(y, sr, n_mfcc=feat_cfg['n_mfcc'])
            fs     = scaler.transform(feat.reshape(1,-1))
            ll_m   = gmm_male.score_samples(fs)[0]
            ll_f   = gmm_female.score_samples(fs)[0]
            label  = 'male' if ll_m > ll_f else 'female'
            conf   = float(1 / (1 + np.exp(-abs(ll_m-ll_f)/10)))
            probs  = {'male': conf if label=='male' else 1-conf,
                      'female': 1-conf if label=='male' else conf}

        elif model_type == 'cnn':
            mel    = _mel_spec(y, sr)
            x_t    = torch.tensor(mel[np.newaxis,np.newaxis,:,:],
                                  dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                out   = cnn_model(x_t)
                probs_t = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs_t))
            label    = le.inverse_transform([pred_idx])[0]
            conf     = float(probs_t[pred_idx])
            probs    = {'female': float(probs_t[0]), 'male': float(probs_t[1])}
        else:
            return {'error': f'Unknown model: {model_type}'}

        return {
            'prediction': label,
            'confidence': round(conf, 4),
            'probabilities': {k: round(v, 4) for k, v in probs.items()},
            'model': model_type,
            'stats': stats,
            'inference_ms': round((time.time()-t0)*1000, 1),
            'demo_mode': False
        }
    except Exception as e:
        return {'error': str(e)}

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           models_loaded=MODELS_LOADED,
                           device=str(DEVICE))

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model', 'cnn')

    # ── Handle recorded audio (base64 blob) ──
    if 'audio_blob' in request.form:
        audio_b64 = request.form['audio_blob']
        # Remove data URI header if present
        if ',' in audio_b64:
            audio_b64 = audio_b64.split(',', 1)[1]
        audio_bytes = base64.b64decode(audio_b64)
        tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'rec_{int(time.time())}.webm')
        with open(tmp_path, 'wb') as f:
            f.write(audio_bytes)
        try:
            result = predict_gender(tmp_path, model_type)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        return jsonify(result)

    # ── Handle uploaded file ──
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'Unsupported format .{ext}'}), 400

    fname = secure_filename(file.filename)
    path  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(path)
    try:
        result = predict_gender(path, model_type)
    finally:
        if os.path.exists(path): os.remove(path)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status':'ok','models_loaded':MODELS_LOADED,'device':str(DEVICE)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
