from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# 유틸: 문자열로 온 값들 안전 변환
def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)

def _as_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _as_int(v, default=None):
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _norm_optimizer(v):
    # "null", "", None 등은 미지정 처리
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
        return None
    return str(v)

def _unique_run_dir(base_project: str, job_id: str) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # runs/yolo-autolabel-trainer/20251113_130102_abcd12  같은 식
    project = base_project  # 예: "runs/yolo-autolabel-trainer"
    name = f"{ts}_{job_id[:6]}"
    return project, name

# core/trainer.py (콜백 내부에 추가)
def _to_jsonable(x):
    # 로컬 import로 선택 의존성 처리
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    # torch.Tensor 처리
    if torch is not None and isinstance(x, torch.Tensor):
        try:
            return x.item() if x.ndim == 0 else x.detach().cpu().tolist()
        except Exception:
            return str(x)

    # numpy 처리
    if np is not None:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()

    # 컬렉션 재귀 처리
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # 기본형
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    # 마지막 수단
    try:
        return float(x)
    except Exception:
        return str(x)


def train_yolo(data_dir: str, out_dir: str, hyper: dict, progress=None, is_ai_training: bool = False) -> dict:
    """
    hyper 예시(모두 선택적):
    {
      "model": "yolo12n.pt",
      "epochs": 1, "imgsz": 640, "batch": 32, "device": "cuda:0",
      "workers": 8, "optimizer": "null", "lr0": 0.01, "lrf": 0.01,
      "weight_decay": 0.0005, "momentum": 0.937, "patience": 30,
      "save": true, "augment": true, "mosaic": true, "mixup": false,
      "job_id": "abc123"  # (권장) run 폴더 고유화에 사용
    }
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model_name = hyper.get("model", "yolo12n.pt")
    model = YOLO(model_name)

    # 공통 기본값
    defaults = {
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": None,
        "workers": 8,
        "optimizer": None,
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "patience": 30,
        "save": True,
        "augment": None,
        "mosaic": None,
        "mixup": None,
    }

    # 입력값 정규화
    epochs       = _as_int(hyper.get("epochs"), defaults["epochs"])
    imgsz        = _as_int(hyper.get("imgsz"), defaults["imgsz"])
    batch        = _as_int(hyper.get("batch"), defaults["batch"])
    device       = hyper.get("device", defaults["device"])
    workers      = _as_int(hyper.get("workers"), defaults["workers"])
    optimizer    = _norm_optimizer(hyper.get("optimizer", defaults["optimizer"]))
    lr0          = _as_float(hyper.get("lr0"), defaults["lr0"])
    lrf          = _as_float(hyper.get("lrf"), defaults["lrf"])
    weight_decay = _as_float(hyper.get("weight_decay"), defaults["weight_decay"])
    momentum     = _as_float(hyper.get("momentum"), defaults["momentum"])
    patience     = _as_int(hyper.get("patience"), defaults["patience"])

    # bool 계열(문자열도 허용)
    save   = defaults["save"] if hyper.get("save")   is None else _as_bool(hyper.get("save"))
    augment= None if hyper.get("augment") is None else _as_bool(hyper.get("augment"))
    mosaic = None if hyper.get("mosaic")  is None else _as_bool(hyper.get("mosaic"))
    mixup  = None if hyper.get("mixup")   is None else _as_bool(hyper.get("mixup"))

    # 고유 run 폴더 구성
    job_id = (hyper.get("job_id") or getattr(progress, "job_id", None) or "nojob")
    project, name = _unique_run_dir(out_dir, job_id)

    # .train 인자 구성
    train_kwargs = {
        "data": str(Path(data_dir, "data.yaml")),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,

        # ✅ 매 실행마다 고유 폴더
        "project": project,     # ex) out_dir
        "name": name,           # ex) 20251113_130102_abcd12

        # ✅ 누적 방지: 동일 폴더 재사용/재개 금지
        "exist_ok": False,
        "resume": False,

        "workers": workers,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "patience": patience,
        "save": save,
    }
    if device is not None:
        train_kwargs["device"] = device
    if optimizer is not None:
        train_kwargs["optimizer"] = optimizer
    if augment is not None:
        train_kwargs["augment"] = augment
    if mosaic is not None:
        train_kwargs["mosaic"] = mosaic
    if mixup is not None:
        train_kwargs["mixup"] = mixup

    # 런 폴더 경로 (학습 전 계산해둠)
    run_dir = Path(project) / name
    best_pt = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"

    # 진행 콜백
    if progress is not None:
        print(f"[trainer] Progress callback 등록: job_id={progress.job_id}")
        
        def _on_fit_epoch_end(trainer):
            print(f"[trainer] on_fit_epoch_end 콜백 실행됨")
            try:
                epoch = int(getattr(trainer, "epoch", 0))
                total_epochs = int(getattr(trainer, "epochs", 0) or getattr(getattr(trainer, "args", None), "epochs", 0))
                print(f"[trainer] Epoch {epoch}/{total_epochs} 완료")
            except Exception as e:
                print(f"[trainer] Epoch 정보 추출 실패: {e}")
                epoch = 0
                total_epochs = 0

            raw_metrics = {}
            m = getattr(trainer, "metrics", None)
            if isinstance(m, dict):
                raw_metrics.update(m)
            elif m is not None:
                raw_metrics["metrics"] = str(m)

            for attr in ("loss", "tloss", "nloss", "lr", "ema_loss"):
                if hasattr(trainer, attr):
                    raw_metrics[attr] = getattr(trainer, attr)

            safe_metrics = _to_jsonable(raw_metrics)
            print(f"[trainer] 메트릭 준비 완료: {list(safe_metrics.keys())}")
            
            try:
                print(f"[trainer] train_log 발행 시도...")
                progress.train_log(epoch=epoch, total_epochs=total_epochs, metrics=safe_metrics)
                print(f"[trainer] ✅ train_log 발행 성공")
                
                print(f"[trainer] train_llm_log 발행 시도...")
                progress.train_llm_log(epoch=epoch, total_epochs=total_epochs, metrics=safe_metrics)
                print(f"[trainer] ✅ train_llm_log 발행 성공")
            except Exception as e:
                print(f"[progress] ❌ train.log publish failed (after sanitize): {e}")
                import traceback
                traceback.print_exc()

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        print(f"[trainer] ✅ Progress callback 등록 완료")

    # 학습 실행
    r = model.train(**train_kwargs)

    # 반환(경로 포함)
    metrics_obj = {}
    
    # 1순위: results.csv에서 최종 epoch 메트릭 읽기
    print(f"[trainer] 🔍 results_csv 경로 확인: {results_csv}")
    print(f"[trainer] 🔍 results_csv 존재 여부: {results_csv.exists()}")
    
    if results_csv.exists():
        try:
            import pandas as pd
            import numpy as np
            print(f"[trainer] 📖 results.csv 읽기 시도...")
            df = pd.read_csv(results_csv)
            print(f"[trainer] 📊 CSV shape: {df.shape}, empty: {df.empty}")
            
            if not df.empty:
                # 마지막 row의 메트릭 추출
                last_row = df.iloc[-1].to_dict()
                print(f"[trainer] 📋 마지막 row 컬럼 수: {len(last_row)}")
                
                # 숫자형 메트릭만 추출 (NaN 제외)
                metrics_obj = {}
                for k, v in last_row.items():
                    try:
                        # NaN이 아니고 숫자로 변환 가능한 값만 추가
                        if pd.notna(v):
                            metrics_obj[k] = float(v)
                    except (ValueError, TypeError):
                        # 숫자로 변환 불가능한 값은 무시
                        pass
                print(f"[trainer] ✅ results.csv에서 메트릭 추출 완료: {len(metrics_obj)}개 메트릭, 키={list(metrics_obj.keys())[:10]}")
            else:
                print(f"[trainer] ⚠️ results.csv가 비어있음")
        except Exception as e:
            print(f"[trainer] ⚠️ results.csv 읽기 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[trainer] ⚠️ results.csv 파일이 존재하지 않음: {results_csv}")
    
    # 2순위: results 객체에서 메트릭 추출 시도
    if not metrics_obj:
        try:
            # results.results_dict 또는 results.metrics 시도
            if hasattr(r, "results_dict") and isinstance(r.results_dict, dict):
                metrics_obj = r.results_dict
            elif hasattr(r, "metrics") and isinstance(r.metrics, dict):
                metrics_obj = r.metrics
            else:
                metrics_obj = getattr(r, "metrics", {}) or {}
        except Exception as e:
            print(f"[trainer] ⚠️ results 객체에서 메트릭 추출 실패: {e}")
            metrics_obj = {}

    return {
        "metrics": metrics_obj,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "results_csv": str(results_csv) if results_csv.exists() else None,
    }
