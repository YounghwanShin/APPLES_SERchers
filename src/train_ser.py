"""IEMOCAP 음성 → Valence/Arousal/Dominance 회귀. Leave-one-session-out 5-fold 교차검증.

실행: python src/train_ser.py --config configs/ser_iemocap.yaml
"""
import argparse
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          get_linear_schedule_with_warmup, set_seed)

# IEMOCAP EmoEvaluation 파일의 차원 라벨 순서([V, A, D])와 같다.
TARGETS = ("valence", "arousal", "dominance")
SESSIONS = (1, 2, 3, 4, 5)
# 예: [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
LABEL_LINE = re.compile(r"^\[\S+ - \S+\]\s+(Ses0([1-5])\S+)\s+\S+\s+\[([\d.]+), ([\d.]+), ([\d.]+)\]")


@dataclass(frozen=True)
class DataConfig:
    iemocap_root: str
    max_train_seconds: float
    valid_ratio: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    gradient_checkpointing: bool


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    warmup_ratio: float
    epochs: int
    patience: int
    bf16: bool
    num_workers: int
    seed: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def from_dict(cls, raw):
    """dict를 dataclass로 변환한다. 키 누락·오타·타입 불일치는 오류로 처리한다."""
    obj = cls(**raw)
    for f in fields(cls):
        value = getattr(obj, f.name)
        if type(value) is not f.type:
            raise TypeError(f"{cls.__name__}.{f.name}: {f.type.__name__} 타입이어야 합니다 (현재 {value!r}).")
    return obj


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sections = {f.name: f.type for f in fields(Config)}
    if set(raw) != set(sections):
        raise ValueError(f"설정 최상위 키는 {sorted(sections)} 이어야 합니다 (현재 {sorted(raw)}).")
    return Config(**{name: from_dict(cls, raw[name]) for name, cls in sections.items()})


def load_metadata(root):
    """EmoEvaluation 요약 줄에서 발화 ID, 세션, 음성 경로, 평가자 평균 V/A/D(1~5)를 읽는다."""
    rows = []
    for path in sorted(root.glob("Session[1-5]/dialog/EmoEvaluation/Ses*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = LABEL_LINE.match(line)
            if match is None:
                continue
            utterance_id, session, *values = match.groups()
            dialog = utterance_id.rsplit("_", 1)[0]
            wav = root / f"Session{session}" / "sentences" / "wav" / dialog / f"{utterance_id}.wav"
            rows.append({"utterance_id": utterance_id, "session": int(session), "path": str(wav),
                         **dict(zip(TARGETS, map(float, values)))})
    df = pd.DataFrame(rows)
    if df.empty or set(df.session) != set(SESSIONS):
        raise FileNotFoundError(f"{root}에서 Session1~5의 EmoEvaluation 라벨을 찾지 못했습니다.")
    if df.utterance_id.duplicated().any():
        raise ValueError("중복된 발화 ID가 있습니다.")
    missing = [p for p in df.path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"음성 파일 {len(missing)}개가 없습니다. 예: {missing[0]}")
    return df


def split_fold(df, test_session, valid_ratio, seed):
    """테스트 세션 하나를 남기고, 나머지 세션 발화에서 검증셋을 무작위 추출한다."""
    test = df[df.session == test_session]
    rest = df[df.session != test_session]
    valid = rest.sample(frac=valid_ratio, random_state=seed)
    return rest.drop(index=valid.index), valid, test


class UtteranceDataset(Dataset):
    """(음성, V/A/D 라벨)을 반환한다. max_samples가 주어지면 그보다 긴 음성을 무작위 구간으로 자른다."""

    def __init__(self, df, sample_rate, max_samples=None):
        self.paths = df.path.tolist()
        self.labels = df[list(TARGETS)].to_numpy(np.float32)
        self.sample_rate = sample_rate
        self.max_samples = max_samples

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        wav, sample_rate = sf.read(self.paths[index], dtype="float32")
        if sample_rate != self.sample_rate or wav.ndim != 1:
            raise ValueError(f"{self.paths[index]}: {self.sample_rate}Hz 모노 음성이어야 합니다 "
                             f"(sample_rate={sample_rate}, shape={wav.shape}).")
        if self.max_samples is not None and len(wav) > self.max_samples:
            # DataLoader 워커마다 시드가 다른 torch 난수를 사용한다.
            start = int(torch.randint(len(wav) - self.max_samples + 1, ()))
            wav = wav[start:start + self.max_samples]
        return wav, self.labels[index]


class Collator:
    """음성을 정규화·패딩하고 (input_values, attention_mask, labels) 배치를 만든다."""

    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, batch):
        wavs, labels = zip(*batch)
        inputs = self.extractor(list(wavs), sampling_rate=self.extractor.sampling_rate, padding=True,
                                return_attention_mask=True, return_tensors="pt")
        return inputs["input_values"], inputs["attention_mask"], torch.from_numpy(np.stack(labels))


def ccc(pred, gold):
    """차원별 일치상관계수(CCC). pred, gold: [N, 3] → [3]"""
    pred_mean, gold_mean = pred.mean(0), gold.mean(0)
    covariance = ((pred - pred_mean) * (gold - gold_mean)).mean(0)
    return 2 * covariance / (pred.var(0, unbiased=False) + gold.var(0, unbiased=False)
                             + (pred_mean - gold_mean) ** 2)


def regression_metrics(pred, gold):
    ccc_values = ccc(pred, gold)
    rmse_values = (pred - gold).pow(2).mean(0).sqrt()
    metrics = {f"ccc_{t}": ccc_values[i].item() for i, t in enumerate(TARGETS)}
    metrics.update({f"rmse_{t}": rmse_values[i].item() for i, t in enumerate(TARGETS)})
    metrics["ccc_mean"] = ccc_values.mean().item()
    return metrics


def subset_metrics(pred, gold, utterance_ids):
    """전체와 함께 즉흥(impro)·대본(script) 발화별 지표를 구한다. 대본은 세션 간 문장이 겹친다."""
    is_script = torch.tensor(utterance_ids.str.contains("_script").to_numpy())
    return {"all": regression_metrics(pred, gold),
            "impro": regression_metrics(pred[~is_script], gold[~is_script]),
            "script": regression_metrics(pred[is_script], gold[is_script])}


def build_model(name, label_mean, gradient_checkpointing):
    """사전학습 인코더 + 평균 풀링 회귀 헤드. CNN 특징 추출기는 고정하고 transformer는 학습한다."""
    model = AutoModelForAudioClassification.from_pretrained(
        name, num_labels=len(TARGETS), id2label=dict(enumerate(TARGETS)),
        label2id={t: i for i, t in enumerate(TARGETS)})
    model.freeze_feature_encoder()
    # 출력 초기값을 학습 라벨 평균(1~5 척도)으로 둔다.
    with torch.no_grad():
        model.classifier.bias.copy_(torch.from_numpy(label_mean))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def predict(model, loader, device, bf16):
    model.eval()
    preds, golds = [], []
    for inputs, mask, labels in tqdm(loader, desc="eval", leave=False):
        with autocast(device, bf16):
            logits = model(inputs.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.float().cpu())
        golds.append(labels)
    return torch.cat(preds).double(), torch.cat(golds).double()


def train_epoch(model, loader, optimizer, scheduler, device, bf16, desc):
    model.train()
    total = 0.0
    progress = tqdm(loader, desc=desc, leave=False)
    for inputs, mask, labels in progress:
        with autocast(device, bf16):
            logits = model(inputs.to(device), attention_mask=mask.to(device)).logits
        loss = (1 - ccc(logits.float(), labels.to(device))).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total / len(loader)


def run_fold(cfg, metadata, test_session, device):
    """한 fold를 학습하고 검증 CCC 최고 시점의 모델로 테스트 세션을 평가한다."""
    fold_dir = Path(cfg.train.output_dir) / f"fold{test_session}"
    result_path = fold_dir / "metrics.json"
    if result_path.exists():
        tqdm.write(f"[fold{test_session}] 저장된 결과를 사용합니다: {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.train.seed)

    train_df, valid_df, test_df = split_fold(metadata, test_session, cfg.data.valid_ratio, cfg.train.seed)
    extractor = AutoFeatureExtractor.from_pretrained(cfg.model.name)
    sample_rate = extractor.sampling_rate
    collate = Collator(extractor)

    def eval_loader(df):
        return DataLoader(UtteranceDataset(df, sample_rate), batch_size=cfg.train.eval_batch_size,
                          num_workers=cfg.train.num_workers, collate_fn=collate)

    # CCC 손실이 배치 통계이므로 크기가 batch_size보다 작은 마지막 배치는 버린다.
    train_loader = DataLoader(
        UtteranceDataset(train_df, sample_rate, int(cfg.data.max_train_seconds * sample_rate)),
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.train.num_workers, collate_fn=collate)
    valid_loader, test_loader = eval_loader(valid_df), eval_loader(test_df)
    tqdm.write(f"[fold{test_session}] train {len(train_df)} / valid {len(valid_df)} / test {len(test_df)} 발화")

    model = build_model(cfg.model.name, train_df[list(TARGETS)].to_numpy(np.float32).mean(0),
                        cfg.model.gradient_checkpointing).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.train.learning_rate)
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.train.warmup_ratio * total_steps), total_steps)

    best_dir = fold_dir / "best_model"
    history, best, epochs_without_improvement = [], None, 0
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, cfg.train.bf16,
                                 desc=f"fold{test_session} epoch {epoch}/{cfg.train.epochs}")
        valid = regression_metrics(*predict(model, valid_loader, device, cfg.train.bf16))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid": valid})
        improved = best is None or valid["ccc_mean"] > best["valid"]["ccc_mean"]
        tqdm.write(f"[fold{test_session}] epoch {epoch}: train_loss {train_loss:.4f} | valid CCC "
                   + " ".join(f"{t[0].upper()} {valid[f'ccc_{t}']:.3f}" for t in TARGETS)
                   + f" | mean {valid['ccc_mean']:.3f}" + (" *" if improved else ""))
        if improved:
            best, epochs_without_improvement = history[-1], 0
            model.save_pretrained(best_dir)
            extractor.save_pretrained(best_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.patience:
                break

    model = AutoModelForAudioClassification.from_pretrained(best_dir).to(device)
    pred, gold = predict(model, test_loader, device, cfg.train.bf16)
    predictions = test_df[["utterance_id", "session", *TARGETS]].copy()
    for i, t in enumerate(TARGETS):
        predictions[f"pred_{t}"] = pred[:, i].numpy()
    predictions.to_csv(fold_dir / "predictions.csv", index=False)

    result = {"test_session": test_session, "best_epoch": best["epoch"], "valid": best["valid"],
              "test": subset_metrics(pred, gold, test_df.utterance_id), "history": history}
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    tqdm.write(f"[fold{test_session}] test CCC mean {result['test']['all']['ccc_mean']:.3f}")
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def summarize(results, out_dir):
    """fold별 테스트 지표와 5-fold 평균·표준편차, 전체 발화 예측을 저장한다."""
    folds = pd.json_normalize([r["test"] for r in results]).set_index(pd.Index(SESSIONS, name="test_session"))
    folds.to_csv(out_dir / "fold_results.csv")
    summary = {"mean": folds.mean().to_dict(), "std": folds.std(ddof=1).to_dict()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat(pd.read_csv(out_dir / f"fold{s}" / "predictions.csv") for s in SESSIONS).to_csv(
        out_dir / "predictions.csv", index=False)
    for subset in ("all", "impro", "script"):
        tqdm.write(f"5-fold 테스트 CCC [{subset}] " + " ".join(
            f"{t}: {summary['mean'][f'{subset}.ccc_{t}']:.3f}±{summary['std'][f'{subset}.ccc_{t}']:.3f}"
            for t in TARGETS))


def main():
    parser = argparse.ArgumentParser(description="IEMOCAP 음성 V/A/D 회귀 5-fold 교차검증")
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 같은 output_dir에 다른 설정의 결과가 섞이지 않도록 한다.
    saved = out_dir / "config.yaml"
    if saved.exists() and yaml.safe_load(saved.read_text(encoding="utf-8")) != asdict(cfg):
        raise ValueError(f"{saved}의 설정과 현재 설정이 다릅니다. 새 실험은 train.output_dir를 바꾸세요.")
    saved.write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(Path(cfg.data.iemocap_root))
    tqdm.write(f"IEMOCAP 발화 {len(metadata)}개, device={device}")
    results = [run_fold(cfg, metadata, session, device) for session in SESSIONS]
    summarize(results, out_dir)


if __name__ == "__main__":
    main()
