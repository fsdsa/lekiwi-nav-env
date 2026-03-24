# VIVA — VLM-Instructed VLA Architecture

## 아키텍처

```
[A100 서버 — 항상 실행]
  vLLM (port 8000)  ─ Qwen2.5-VL-7B-Instruct, ~19.8GB VRAM
  VLA  (port 8002)  ─ Pi0-FAST 2.9B via LeRobot 0.5.0, ~7.7GB VRAM
                      합계 ~27.5GB / 40GB

[3090 로컬 — 유저가 실행]
  run_full_task.py  ─ Isaac Sim + 카메라 + 제어 루프
```

### 4-Skill Pipeline

```
S1: navigate          — source object 탐색 이동 (arm all-zero, gripper open, base 이동)
S2: approach & lift   — 정밀 접근 + 파지 + 들어올리기 (arm + gripper + base)
S3: carry             — 물체 들고 dest object까지 이동 (arm lifted pose, gripper closed, base 이동)
S4: approach & place  — 정밀 접근 + 배치 + rest pose 복귀 (arm + gripper + base)
```

- S1/S3 = 대략적 이동 (VLM이 방향 지시, VLA가 실행)
- S2/S4 = 정밀 제어 (VLM이 상황 판단, VLA가 arm+gripper+base 동시 제어)

### 실행 흐름 (VIVA mode)

```
1. 유저 지시어 입력
   "약병을 찾아서 빨간 컵 옆에 놓아"
        │
2. VLM /classify (text-only, 1회, ~250ms)
   → {"source": "medicine bottle", "dest": "red cup"}
        │
3. Isaac Sim 환경 로드 + 카메라 초기화
        │
4. 메인 루프 (6.4 Hz)
   │
   ├─ 매 step: robot_status 텍스트 생성 (joint position, gripper, contact, depth)
   │
   ├─ VLM (비동기 ~191ms):
   │    base_cam RGB + robot_status → 스킬별 프롬프트 → instruction
   │    S1: "move forward" / "turn left" / "TARGET_FOUND"
   │    S2: "move toward the medicine bottle" / "LIFTED_COMPLETE" / "OBSTACLE"
   │    S3: "carry medicine bottle and move forward" / "TARGET_FOUND"
   │    S4: "place the medicine bottle next to the red cup" / "PLACE_COMPLETE"
   │
   ├─ VLA (동기):
   │    base_cam RGB + wrist_cam RGB + 9D state + instruction → 9D action
   │
   ├─ Safety layer (스킬별):
   │    S1/S3: depth < 0.3m → base vx,vy 정지 (wz 유지)
   │    S2/S4: 비활성화 (depth_warning을 VLM 텍스트로 전달)
   │
   ├─ 스킬 전환 감지 → VLA action buffer 리셋
   │
   └─ env.step(action)
```

### 스킬 전환 조건

| 전환 | 조건 | 판단 주체 |
|------|------|-----------|
| S1 → S2 | VLM이 source object 발견 → "TARGET_FOUND" | VLM |
| S2 → S3 | lifted pose + contact → VLM에 "LIFTED" → "LIFTED_COMPLETE" | VLM |
| S3 → S4 | VLM이 dest object 발견 → "TARGET_FOUND" | VLM |
| S4 → DONE | VLM이 배치 완료 → "PLACE_COMPLETE" | VLM |

### 장애물 회피

S2/S4에서 depth_warning이 VLM에 전달되고, VLM이 "OBSTACLE" 출력 시:
- S2 → S1(navigate)로 전환 → TARGET_FOUND → S2 복귀
- S4 → S3(carry)로 전환 → TARGET_FOUND → S4 복귀

## 파일 구조

| 파일 | 위치 | 역할 |
|------|------|------|
| `run_full_task.py` | 로컬 | 메인 실행 (VIVA/single_vla 모드 분기) |
| `vlm_orchestrator.py` | 로컬 | classify + RelativePlacementOrchestrator(레거시) + VIVAOrchestrator(4-skill) |
| `vlm_prompts.py` | 로컬 | CLASSIFY / INSTRUCT(레거시) / VIVA S1~S4 프롬프트 |
| `vla_inference_server.py` | 서버 | Pi0-FAST FastAPI 서버 |
| `launch_vlm_server.sh` | 서버 | vLLM 시작 스크립트 |
| `test_roundtrip.py` | 로컬 | 레이턴시 측정용 (Hz 벤치마크) |
| `run_vlm_navigate.py` | 로컬 | Navigate 전용 (BC+VLM, 전체 파이프라인과 별개) |

## 실행 모드

| 모드 | `--mode` | 설명 |
|------|----------|------|
| VIVA | `viva` (기본) | VLM이 4-skill 상태 머신 기반으로 스킬별 프롬프트 사용 |
| 비교군 ①-B | `single_vla` | 단일 VLA (VLM 없음), RelativePlacementOrchestrator 사용 |

## 사용 방법

### 1. 서버 준비 (A100, 최초 1회)

```bash
ssh -i ~/.ssh/private.pem jovyan@218.148.55.186 -p 30179

# VLM 서버
conda activate vllm
bash launch_vlm_server.sh  # port 8000

# VLA 서버 (별도 터미널)
conda activate lerobotpi0v2
python vla_inference_server.py --port 8002
```

### 2. SSH 터널 (로컬)

```bash
ssh -f -N \
  -L 8000:localhost:8000 \
  -L 8002:localhost:8002 \
  -i ~/.ssh/private.pem \
  jovyan@218.148.55.186 -p 30179
```

### 3. 전체 파이프라인 실행 (로컬 3090)

```bash
# VIVA mode (기본)
PYTHONUNBUFFERED=1 python vllm/run_full_task.py \
  --user_command "find the medicine bottle and place it next to the red cup" \
  --object_usd <source.usd> \
  --mode viva \
  --headless

# 비교군 ①-B (single VLA)
PYTHONUNBUFFERED=1 python vllm/run_full_task.py \
  --user_command "find the medicine bottle and place it next to the red cup" \
  --object_usd <source.usd> \
  --mode single_vla \
  --headless
```

주요 인자:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `viva` | `viva` (VIVA 4-skill) / `single_vla` (비교군 ①-B) |
| `--user_command` | "find the medicine bottle..." | 유저 지시어 |
| `--target_object` | (없음) | 수동 지정 시 VLM classify 건너뜀 |
| `--dest_object` | (없음) | 수동 지정 시 VLM classify 건너뜀 |
| `--navigate_timeout` | 2000 | S1 navigate timeout (steps) |
| `--approach_lift_timeout` | 1000 | S2 approach & lift timeout |
| `--carry_timeout` | 2000 | S3 carry timeout |
| `--approach_place_timeout` | 1000 | S4 approach & place timeout |
| `--max_total_steps` | 6000 | 전체 최대 스텝 (~10분) |
| `--safety_dist` | 0.3 | depth 긴급정지 거리 (m) |

## 검증 결과 (2026-03-10, lerobot 0.5.0)

### 레이턴시

| 구간 | 평균 | 비고 |
|------|------|------|
| VLM (vLLM Qwen2.5-VL) | 191ms | 비동기, ~86회/200step |
| VLA (Pi0-FAST) | 27-58ms | 동기 |
| 전체 루프 | ~156ms | **6.4 Hz** |

### GPU 메모리 (A100 40GB)

```
VLM (vLLM Qwen2.5-VL-7B): ~19.8GB  (--gpu-memory-utilization 0.50)
VLA (Pi0-FAST 2.9B):       ~7.7GB
합계:                      ~27.5GB / 40GB
```

## 서버 환경 (lerobotpi0v2)

```
Python 3.12, conda env: lerobotpi0v2
lerobot 0.5.0 (git source v0.5.0 tag + [pi] extra)
transformers 5.3.0, torch 2.10.0+cu128
fastapi 0.135.1, uvicorn 0.41.0
```

## 주의사항

1. **attention mask `.bool()`**: `vla_inference_server.py` line 133에서 처리
2. **`validate_action_token_prefix = False`**: pretrained base 모델은 garbage output 시 assert
3. **HuggingFace 로그인 필요**: `huggingface-cli login` (google/paligemma-3b-pt-224 gated repo)
4. **`PYTHONUNBUFFERED=1`**: stdout 버퍼링 방지 (로그 실시간 출력)
5. **스킬 전환 시 VLA buffer 리셋**: 특히 S2→S3 (arm action chunk 잔류 → 물체 낙하 위험)
6. **S2/S4 safety layer 비활성화**: depth_warning을 VLM 텍스트로 전달 (안 끄면 물체 앞에서 멈춤)
7. **`get_contact_detected()`**: 환경의 contact sensor API에 의존 (jaw/wrist force > 1.0N)
