# VLM + VLA 전체 파이프라인

## 아키텍처

```
[A100 서버 — 항상 실행]
  vLLM (port 8000)  ─ Qwen2.5-VL-7B-Instruct, ~19.8GB VRAM
  VLA  (port 8002)  ─ Pi0-FAST 2.9B via LeRobot, ~5.9GB VRAM
                      합계 ~26GB / 40GB

[3090 로컬 — 유저가 실행]
  run_full_task.py  ─ Isaac Sim + 카메라 + 제어 루프
```

### 실행 흐름

```
1. 유저 지시어 입력
   "약병을 찾아서 빨간 컵 옆에 놓아"
        │
2. VLM /classify (text-only, 1회, ~250ms)
   → {"source": "medicine bottle", "dest": "red cup"}
        │
3. Isaac Sim 환경 로드 + 카메라 초기화
        │
4. 메인 루프 (6.5 Hz)
   │
   ├─ 매 30 step (~3초, VLM 비동기):
   │    base_cam RGB ──HTTP──→ A100 VLM → 자연어 instruction
   │    "turn right slowly to search for the medicine bottle"
   │    "approach and pick up the medicine bottle"
   │    "move toward the red cup and place the medicine bottle next to it"
   │    "done"
   │
   ├─ 매 step (VLA 동기):
   │    base_cam RGB + wrist_cam RGB + 9D state + instruction
   │    ──HTTP──→ A100 VLA → 9D action [arm5, gripper1, base3]
   │
   ├─ depth safety (로컬 처리, 서버 전송 안 함):
   │    base_cam depth → 전방 중앙 1/3 min depth < 0.3m → base 정지 (회전은 허용)
   │
   └─ env.step(action)
```

## 파일 구조

| 파일 | 위치 | 역할 |
|------|------|------|
| `run_full_task.py` | 로컬 | 메인 실행 스크립트 (전체 파이프라인) |
| `vlm_orchestrator.py` | 로컬 | classify_user_request() + RelativePlacementOrchestrator |
| `vlm_prompts.py` | 로컬 | CLASSIFY / INSTRUCT / NAVIGATE 프롬프트 |
| `vla_inference_server.py` | 서버 | Pi0-FAST FastAPI 서버 |
| `launch_vlm_server.sh` | 서버 | vLLM 시작 스크립트 |
| `test_roundtrip.py` | 로컬 | 레이턴시 측정용 (Hz 벤치마크) |
| `test_vlm_vla.py` | 로컬 | VLM+VLA 단순 API 테스트 (Isaac Sim 없이) |
| `run_vlm_navigate.py` | 로컬 | Navigate 전용 (BC+VLM, 전체 파이프라인과 별개) |

## 사용 방법

### 1. 서버 준비 (A100, 최초 1회)

```bash
# SSH 접속
ssh -i ~/.ssh/private.pem jovyan@218.148.55.186 -p 30179

# VLM 서버 시작
conda activate vllm
bash launch_vlm_server.sh
# → port 8000, Qwen2.5-VL-7B, --gpu-memory-utilization 0.50

# VLA 서버 시작 (별도 터미널)
conda activate lerobotpi0
python vla_inference_server.py --port 8002
# → Pi0-FAST base model (fine-tuned 모델은 --model <path>)
```

### 2. SSH 터널 (로컬, 서버 접근용)

```bash
ssh -f -N \
  -L 8000:localhost:8000 \
  -L 8002:localhost:8002 \
  -i ~/.ssh/private.pem \
  jovyan@218.148.55.186 -p 30179
```

### 3. 전체 파이프라인 실행 (로컬 3090)

```bash
conda activate env_isaaclab
PYTHONPATH="vllm:.:$PYTHONPATH" python vllm/run_full_task.py \
  --user_command "find the medicine bottle and place it next to the red cup" \
  --object_usd <source_object.usd> \
  --headless
```

주요 인자:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--user_command` | "find the medicine bottle..." | 유저 지시어 (VLM이 source/dest 자동 추출) |
| `--target_object` | (없음) | 수동 지정 시 VLM classify 건너뜀 |
| `--dest_object` | (없음) | 수동 지정 시 VLM classify 건너뜀 |
| `--vlm_server` | http://localhost:8000 | vLLM 서버 URL |
| `--vla_server` | http://localhost:8002 | VLA 서버 URL |
| `--vlm_interval` | 30 | VLM 호출 간격 (steps, 30 = ~3초) |
| `--max_total_steps` | 6000 | 최대 실행 (6000 = ~10분) |
| `--camera_width` | 640 | 카메라 해상도 |
| `--camera_height` | 400 | 카메라 해상도 |
| `--safety_dist` | 0.3 | depth 긴급정지 거리 (m) |
| `--object_usd` | (없음) | source 물체 USD |
| `--dest_object_usd` | (없음) | destination 물체 USD |

### 4. 레이턴시 벤치마크

```bash
PYTHONPATH="vllm:.:$PYTHONPATH" python vllm/test_roundtrip.py \
  --num_steps 20 --headless
```

### 5. API 단순 테스트 (Isaac Sim 없이)

```bash
cd vllm && python test_vlm_vla.py
```

## 검증 결과 (2026-03-10)

### 레이턴시

| 구간 | 평균 | Hz |
|------|------|------|
| Camera capture | 82ms | — |
| JPEG encode | 1.3ms | — |
| VLM (vLLM Qwen2.5-VL) | 210ms | ~5 Hz |
| VLA (Pi0-FAST) | 22ms (서버 9ms) | ~45 Hz |
| 전체 루프 | 153ms | **6.5 Hz** |

### 동작 확인

- VLM classify: 한국어/영어 모두 정확 추출 (source+dest)
- VLM instruct: 상황별 자연어 instruction 생성 ("turn right slowly to search...", "move forward to find...")
- VLA: base+wrist cam + instruction → 32D action → 9D truncate
- Depth safety: 0.3m 이내 장애물 감지 → base 정지
- 전체 통합: 200 steps (30초) 완주, VLM 7회 호출, safety 47회 트리거

### GPU 메모리 (A100 40GB)

```
VLM (vLLM Qwen2.5-VL-7B): ~19.8GB  (--gpu-memory-utilization 0.50)
VLA (Pi0-FAST 2.9B):       ~5.9GB
합계:                      ~25.9GB / 40GB
```

## 실제 테스트 시 변경할 것

1. **`--object_usd`**: 실제 물체 USD 경로
2. **`--dest_object_usd`**: destination 물체 USD 경로
3. **VLA 모델**: fine-tuned Pi0 체크포인트 (`vla_inference_server.py --model <path>`)
4. **`--user_command`**: 실제 태스크 지시어

## 필수 패치 (서버 lerobotpi0 환경)

Pi0-FAST 실행 시 필요한 패치 2개:

1. **lerobot attention mask**: `vla_inference_server.py` 내 monkey-patch 포함
2. **action token prefix validation**: `config.validate_action_token_prefix = False`

HuggingFace 로그인 필요: `huggingface-cli login` (google/paligemma-3b-pt-224 gated repo)
