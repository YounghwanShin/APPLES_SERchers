"""IEMOCAP A/V/D regression. Transformers 4.57.1; single-GPU Runpod."""
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, Wav2Vec2Model, Wav2Vec2PreTrainedModel,
    Wav2Vec2FeatureExtractor, Trainer, TrainingArguments, EarlyStoppingCallback,
    TrainerCallback, set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput

TARGETS = ['arousal', 'valence', 'dominance']
MODEL_ID = 'facebook/wav2vec2-large-xlsr-53'
FOLDS = {1: ('123', 4), 2: ('124', 3), 3: ('134', 2), 4: ('234', 1)}

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def ccc_metrics(gold, pred):
    gold, pred = np.asarray(gold, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if gold.shape != pred.shape or gold.ndim != 2 or gold.shape[1] != 3:
        raise ValueError('CCC 입력은 동일한 [발화 수, 3] 배열이어야 합니다.')
    if len(gold) < 2 or not np.isfinite(gold).all() or not np.isfinite(pred).all():
        raise ValueError('평가 데이터 수 또는 NaN/Inf를 확인하세요.')
    gm, pm = gold.mean(0), pred.mean(0)
    cov = ((gold-gm)*(pred-pm)).mean(0)
    den = gold.var(0) + pred.var(0) + (gm-pm)**2
    # 두 배열 모두 같은 상수이면 CCC는 정의되지 않아 보수적으로 0 처리합니다.
    ccc = np.divide(2*cov, den, out=np.zeros(3), where=den > 1e-12)
    result = {f'ccc_{name}': float(ccc[i]) for i, name in enumerate(TARGETS)}
    result.update(ccc_mean=float(ccc.mean()), mse=float(np.mean((gold-pred)**2)))
    return result

def read_table(path):
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    pcol = 'path' if 'path' in df else 'audio_path'
    if pcol not in df:
        raise ValueError(f'{path}: path 또는 audio_path 열이 없습니다: {list(df)}')
    if all(x+'_norm' in df for x in TARGETS):
        cols = [x+'_norm' for x in TARGETS]
    elif all(x in df for x in TARGETS):
        cols = TARGETS
    else:
        raise ValueError(f'{path}: A/V/D 라벨 열을 찾을 수 없습니다.')
    out = pd.DataFrame({'path': df[pcol].astype(str).str.strip()})
    for name, col in zip(TARGETS, cols):
        out[name] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='raise')
    vals = out[TARGETS].to_numpy()
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise ValueError(f'{path}: 라벨은 이미 0~1이어야 합니다. 자동 재정규화하지 않습니다.')
    out['utterance_id'] = out.path.map(lambda x: Path(x.replace('\\', '/')).stem)
    out['session'] = out.utterance_id.str.extract(r'^Ses0([1-5])[FM]', expand=False)
    if out.session.isna().any() or out.utterance_id.duplicated().any():
        raise ValueError(f'{path}: 발화 ID의 세션 또는 중복을 확인하세요.')
    out['session'] = out.session.astype(int)
    return out

def load_fold(base, fold):
    tr, va = FOLDS[fold]
    root = Path(base)/'IEMOCAP_preprocessed'
    a = root/'cv_4fold'/f'fold{fold}_train_S{tr}_AVD.csv'
    b = root/'cv_4fold'/f'fold{fold}_val_S{va}_AVD.csv'
    def session_table(s):
        matches = list(root.rglob(f'iemocap_session{s}_preprocessed_AVD.csv'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Session{s} CSV를 한 개로 특정할 수 없습니다: {matches}')
        return read_table(matches[0])
    train = read_table(a) if a.exists() else pd.concat([session_table(int(s)) for s in tr], ignore_index=True)
    val = read_table(b) if b.exists() else session_table(va)
    if set(train.session) != set(map(int, tr)) or set(val.session) != {va}:
        raise ValueError('CSV의 실제 세션이 fold 설정과 다릅니다. Session5는 학습/검증에 넣지 않습니다.')
    if set(train.utterance_id) & set(val.utterance_id):
        raise ValueError('학습/검증 발화가 중복됩니다.')
    return train.reset_index(drop=True), val.reset_index(drop=True)

def cache_audio(df, base, cache_dir):
    """Only requested split audio is read; .npy cache is rebuilt after runtime reset."""
    df = df.copy()
    root = Path(base)/'IEMOCAP_extracted'/'IEMOCAP_full_release'
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    def convert(row):
        p = Path(row.path)
        if not p.is_file():
            uid = row.utterance_id
            p = root/f'Session{row.session}'/'sentences'/'wav'/uid.rsplit('_', 1)[0]/(uid+'.wav')
        if not p.is_file():
            raise FileNotFoundError(f'음성이 없습니다: {row.path}\n확인한 대체 경로: {p}')
        stat = p.stat()
        key = hashlib.sha256(f'v1|{p}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()
        dest = cache_dir/(key+'.npy')
        if not dest.exists():
            audio, rate = sf.read(p, dtype='float32', always_2d=True)
            audio = audio.mean(1)
            if rate != 16000:
                common = math.gcd(int(rate), 16000)
                audio = resample_poly(audio, 16000//common, int(rate)//common).astype('float32')
            if len(audio) == 0 or not np.isfinite(audio).all():
                raise ValueError(f'빈 음성 또는 NaN/Inf: {p}')
            tmp = dest.with_suffix('.tmp')
            with open(tmp, 'wb') as f:
                np.save(f, audio)
            os.replace(tmp, dest)
        audio = np.load(dest, mmap_mode='r')
        return str(dest), len(audio)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(tqdm(pool.map(convert, df.itertuples()), total=len(df), desc='음성 확인/로컬 캐시'))
    df['cache_path'], df['n_samples'] = zip(*result)
    return df

class AudioDataset(Dataset):
    def __init__(self, df, extractor, seconds=12, training=False):
        self.df, self.extractor, self.training = df.reset_index(drop=True), extractor, training
        self.limit = int(seconds*16000)
        self.entries = []
        for i, n in enumerate(self.df.n_samples):
            if training:
                self.entries.append((i, 0, int(n)))
            else:
                self.entries.extend((i, start, min(int(n), start+self.limit)) for start in range(0, int(n), self.limit))
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        row_i, start, end = self.entries[index]
        row = self.df.iloc[row_i]
        audio = np.load(row.cache_path, mmap_mode='r')
        if self.training and end > self.limit:
            start = np.random.randint(0, end-self.limit+1)
            end = start+self.limit
        x = np.array(audio[start:end], dtype=np.float32, copy=True)
        # CNN receptive field requires at least 400 input samples.
        if len(x) < 400:
            x = np.pad(x, (0, 400-len(x)))
        x = self.extractor(x, sampling_rate=16000).input_values[0]
        return {'input_values': x, 'labels': row[TARGETS].to_numpy(dtype=np.float32)}
    def aggregate(self, predictions):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if len(predictions) != len(self.entries):
            raise ValueError('청크 예측 수가 일치하지 않습니다.')
        sums, weights = np.zeros((len(self.df), 3)), np.zeros(len(self.df))
        for pred, (row, start, end) in zip(predictions, self.entries):
            weight = end-start
            sums[row] += pred*weight
            weights[row] += weight
        return sums/weights[:, None]
    def metrics(self, output):
        return ccc_metrics(self.df[TARGETS].to_numpy(), self.aggregate(output.predictions))

class Collator:
    def __init__(self, extractor):
        self.extractor = extractor
    def __call__(self, rows):
        batch = self.extractor.pad([{'input_values': r['input_values']} for r in rows],
            padding=True, return_attention_mask=True, return_tensors='pt')
        batch['labels'] = torch.as_tensor(np.stack([r['labels'] for r in rows]), dtype=torch.float32)
        return batch

class AVDModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = nn.Sequential(nn.Dropout(config.final_dropout),
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh(),
            nn.Dropout(config.final_dropout), nn.Linear(config.hidden_size, 3))
        self.post_init()
    def forward(self, input_values, attention_mask=None, labels=None):
        hidden = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True).last_hidden_state
        mask = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        # float32 accumulation avoids half-precision overflow on longer sequences.
        mask = mask.unsqueeze(-1).float()
        pooled = (hidden.float()*mask).sum(1)/mask.sum(1).clamp_min(1)
        pred = torch.sigmoid(self.classifier(pooled)).float()
        loss = None
        if labels is not None:
            gold = labels.float()
            mse = nn.functional.mse_loss(pred, gold)
            if len(gold) < 2:
                loss = mse
            else:
                gm, pm = gold.mean(0), pred.mean(0)
                cov = ((gold-gm)*(pred-pm)).mean(0)
                den = gold.var(0, unbiased=False)+pred.var(0, unbiased=False)+(gm-pm).square()
                ccc = 2*cov/den.clamp_min(1e-8)
                loss = (1-ccc).mean()+self.config.mse_weight*mse
        if loss is not None and not torch.isfinite(loss).item():
            raise FloatingPointError('학습 손실에 NaN/Inf 발생. 학습을 중단합니다.')
        return SequenceClassifierOutput(loss=loss, logits=pred)

def freeze_layers(model, last_n):
    layers = model.wav2vec2.encoder.layers
    if not 0 <= last_n <= len(layers):
        raise ValueError(f'train_last_n_layers는 0~{len(layers)} 사이여야 합니다.')
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    if last_n:
        for layer in layers[-last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.wav2vec2.encoder.layer_norm.parameters():
            param.requires_grad = True
    # Frozen convolution outputs must not force the entire frozen prefix to retain a graph.
    model.wav2vec2.feature_extractor._requires_grad = False

class CompleteCheckpoint(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        (Path(args.output_dir)/f'checkpoint-{state.global_step}'/'COMPLETE').touch()

def latest_complete(out):
    paths = sorted(Path(out).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    for p in paths:
        required = ['COMPLETE', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']
        if all((p/f).exists() for f in required) and (p/'model.safetensors').exists():
            return str(p)
    if paths:
        raise RuntimeError('완전한 체크포인트가 없습니다. 폴더를 지우지 말고 오류 화면을 확인하세요.')
    return None

def save_predictions(dataset, pred, out):
    df = dataset.df[['utterance_id', 'path']+TARGETS].copy()
    for i, name in enumerate(TARGETS):
        df['pred_'+name] = pred[:, i]
    df.to_csv(out, index=False)

def run_fold(cfg, fold):
    if cfg.get('init_model_path') and fold != 1:
        raise ValueError('이 복구 설정은 Fold1 전용입니다.')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 연결과 PyTorch CUDA 버전을 확인하세요.')
    set_seed(cfg['seed'])
    train, val = load_fold(cfg['base'], fold)
    out = Path(cfg['base'])/'training_runs'/cfg['run_name']/f'fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(cfg, fold=fold)
    signature['data_hash'] = hashlib.sha256((train.to_csv(index=False)+val.to_csv(index=False)).encode()).hexdigest()
    manifest = out/'run_config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('같은 저장 폴더의 설정/데이터가 바뀌었습니다. 새 실험은 run_name을 바꾸세요.')
    write_json(manifest, signature)
    if (out/'validation_metrics.json').exists() and (out/'best_model'/'model.safetensors').exists():
        print(f'Fold {fold}: 완료 결과를 사용합니다.')
        return json.loads((out/'validation_metrics.json').read_text())
    checkpoint = latest_complete(out)
    train = cache_audio(train, cfg['base'], cfg['cache_dir'])
    val = cache_audio(val, cfg['base'], cfg['cache_dir'])
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, return_attention_mask=True)
    ds_train = AudioDataset(train, extractor, cfg['max_seconds'], training=True)
    ds_val = AudioDataset(val, extractor, cfg['max_seconds'])
    print(f'Fold {fold}: train {len(train)}, val {len(val)} 발화 / val {len(ds_val)} 청크')
    print('학습 중 무작위 부분 구간을 사용하는 긴 발화:', int((train.n_samples > cfg['max_seconds']*16000).sum()))
    if checkpoint:
        model = AVDModel.from_pretrained(checkpoint)
        print('이어서 학습:', checkpoint)
    else:
        config = AutoConfig.from_pretrained(MODEL_ID)
        config.num_labels = 3
        config.id2label = dict(enumerate(TARGETS))
        config.label2id = {v:k for k,v in enumerate(TARGETS)}
        config.mse_weight = cfg['mse_weight']
        config.final_dropout = 0.1
        config.layerdrop = 0.0
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        config._attn_implementation = 'eager'
        model = AVDModel.from_pretrained(cfg.get('init_model_path', MODEL_ID), config=config)
    freeze_layers(model, cfg['train_last_n_layers'])
    print('학습 파라미터:', f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    # Validation/checkpoint twice per epoch, approximately; never under 1 step.
    epoch_steps = math.ceil(math.ceil(len(ds_train)/cfg['batch_size'])/cfg['grad_accum'])
    interval = max(50, (epoch_steps//2//25)*25)
    args = TrainingArguments(output_dir=str(out),
        num_train_epochs=cfg['epochs'], per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'], gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['learning_rate'], weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type='linear', fp16=False, bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', save_strategy='steps', eval_steps=50, save_steps=50,
        logging_steps=10, save_total_limit=2, save_safetensors=True,
        load_best_model_at_end=True, metric_for_best_model='ccc_mean', greater_is_better=True,
        dataloader_num_workers=0, dataloader_drop_last=False, remove_unused_columns=False,
        eval_accumulation_steps=8, report_to='none', optim='adamw_torch',
        seed=cfg['seed'], data_seed=cfg['seed'], max_grad_norm=1.0,
        restore_callback_states_from_checkpoint=True)
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=Collator(extractor), processing_class=extractor, compute_metrics=ds_val.metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=max(4, math.ceil(2*epoch_steps/50)), early_stopping_threshold=0.001), CompleteCheckpoint()])
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(out/'best_model'))
        extractor.save_pretrained(out/'best_model')
        prediction = trainer.predict(ds_val)
        pred = ds_val.aggregate(prediction.predictions)
        metrics = ccc_metrics(val[TARGETS].to_numpy(), pred)
        metrics.update(fold=fold, val_session=FOLDS[fold][1], val_utterances=len(val),
            best_checkpoint=trainer.state.best_model_checkpoint, completed_epoch=trainer.state.epoch)
        save_predictions(ds_val, pred, out/'validation_predictions.csv')
        write_json(out/'training_history.json', trainer.state.log_history)
        write_json(out/'validation_metrics.json', metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return metrics
    except torch.cuda.OutOfMemoryError:
        print('GPU 메모리 부족: 새 run_name에서 batch_size=2, grad_accum=8로 시작하세요. 기존 결과는 보존됩니다.')
        raise
    finally:
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

def summarize(cfg):
    root = Path(cfg['base'])/'training_runs'/cfg['run_name']
    rows = [json.loads(p.read_text()) for p in sorted(root.glob('fold*/validation_metrics.json'))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(root/'fold_results.csv', index=False)
    if len(rows) == 4:
        cols = ['ccc_'+x for x in TARGETS]+['ccc_mean']
        write_json(root/'cv_summary.json', {'folds':4, 'mean':df[cols].mean().to_dict(), 'std':df[cols].std(ddof=1).to_dict()})
        print('4-fold 평균:', df[cols].mean().to_dict())
    else:
        print(f'{len(rows)}/4 fold 완료. 아직 4-fold 평균 성능이 아닙니다.')
    return df

def final_test(cfg):
    """Final locked protocol: arithmetic mean of four validation-selected fold models."""
    root = Path(cfg['base'])/'training_runs'/cfg['run_name']
    if not all((root/f'fold{i}'/'validation_metrics.json').exists() for i in FOLDS):
        raise RuntimeError('4개 fold를 모두 완료한 후 최종 테스트를 실행하세요.')
    existing = root/'session5_ensemble_metrics.json'
    if existing.exists():
        print('이미 저장된 최종 테스트 결과:', existing.read_text())
        return json.loads(existing.read_text())
    files = list((Path(cfg['base'])/'IEMOCAP_preprocessed').rglob('iemocap_session5_preprocessed_AVD.csv'))
    if len(files) != 1:
        raise FileNotFoundError(f'Session5 CSV를 한 개로 특정할 수 없습니다: {files}')
    df = read_table(files[0])
    if set(df.session) != {5}:
        raise ValueError('테스트에는 Session5만 사용합니다.')
    df = cache_audio(df, cfg['base'], cfg['cache_dir'])
    predictions = []
    for fold in FOLDS:
        path = root/f'fold{fold}'/'best_model'
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(path)
        ds = AudioDataset(df, extractor, cfg['max_seconds'])
        model = AVDModel.from_pretrained(path)
        args = TrainingArguments(output_dir=str(root/'test_temp'),
            per_device_eval_batch_size=cfg['eval_batch_size'], fp16=False, bf16=False,
            report_to='none', remove_unused_columns=False, eval_accumulation_steps=8)
        trainer = Trainer(model=model, args=args, data_collator=Collator(extractor))
        pred = ds.aggregate(trainer.predict(ds).predictions)
        predictions.append(pred)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    mean_pred = np.mean(predictions, axis=0)
    metrics = ccc_metrics(df[TARGETS].to_numpy(), mean_pred)
    metrics.update(test_session=5, utterances=len(df), protocol='equal-weight ensemble of four fold-best models')
    save_predictions(ds, mean_pred, root/'session5_ensemble_predictions.csv')
    write_json(existing, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description='APPLES 음성 감정 모델 실행')
    parser.add_argument('action', choices=['check', 'train', 'test'])
    parser.add_argument('--project', type=Path, default=Path('/workspace/APPLES_SERchers'))
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    run_name = 'avd_t4_last4_cccmse_v1_fp32_recovery'
    run = project / 'training_runs' / run_name
    saved = json.loads((run/'fold1'/'run_config.json').read_text())
    cfg = {k:v for k,v in saved.items() if k not in ('fold','data_hash')}
    # 기존 체크포인트의 절대 경로와 설정 비교를 모두 보존합니다.
    old = Path('/content/drive/MyDrive/APPLES_SERchers')
    if cfg['base'] != str(old) or cfg['run_name'] != run_name or saved['fold'] != 1:
        raise ValueError('올려준 노트북과 다른 설정입니다. 복구 폴더를 확인하세요.')
    for fold in FOLDS:
        train, val = load_fold(project, fold)
        signature = hashlib.sha256((train.to_csv(index=False)+val.to_csv(index=False)).encode()).hexdigest()
        if fold == 1 and signature != saved['data_hash']:
            raise ValueError('Fold1 CSV 내용이나 순서가 이전 학습과 다릅니다.')
        for row in pd.concat([train,val]).itertuples():
            wav = project/'IEMOCAP_extracted'/'IEMOCAP_full_release'/f'Session{row.session}'/'sentences'/'wav'/row.utterance_id.rsplit('_',1)[0]/(row.utterance_id+'.wav')
            if not wav.is_file() and not Path(row.path).is_file():
                raise FileNotFoundError(wav)
        print(f'Fold {fold}: 학습 {len(train)}, 검증 {len(val)} 발화')
    fold1 = run/'fold1'
    completed = (fold1/'validation_metrics.json').is_file() and (fold1/'best_model'/'model.safetensors').is_file()
    if not completed:
        checkpoint = latest_complete(fold1)
        if not checkpoint:
            raise FileNotFoundError('fold1의 완전한 checkpoint-* 폴더가 필요합니다.')
        state = json.loads((Path(checkpoint)/'trainer_state.json').read_text())
        best = state.get('best_model_checkpoint')
        if best and not (fold1/Path(best).name/'model.safetensors').is_file():
            raise FileNotFoundError(f'기존 최고 체크포인트도 복사하세요: {best}')
        print('Fold1 재개 지점:', checkpoint, 'step', state['global_step'])
    if args.action == 'check':
        print('파일 검사 완료. 학습은 시작하지 않았습니다.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 연결을 확인하세요.')
    if old != project:
        old.parent.mkdir(parents=True, exist_ok=True)
        if not old.exists() and not old.is_symlink():
            old.symlink_to(project, target_is_directory=True)
        if old.resolve() != project:
            raise RuntimeError(f'{old}가 다른 위치를 가리킵니다. 기존 파일을 보존하고 중단합니다.')
    if args.action == 'test':
        final_test(cfg)
        return
    for fold in FOLDS:
        fold_cfg = cfg.copy()
        if fold != 1:
            fold_cfg.pop('init_model_path', None)
        run_fold(fold_cfg, fold)
    summarize(cfg)

if __name__ == '__main__':
    main()
