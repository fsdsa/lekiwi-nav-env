# VIVA 파이프라인 검증 체크리스트

base 모델(fine-tune 전)로 Isaac Sim 실행하여 파이프라인 로직이 설계대로 흘러가는지 검증.
VLA action은 의미 없으므로, **VLM 호출 로직 / 스킬 전환 / safety layer / instruction 고정** 등 제어 흐름을 확인한다.

현실적으로 base 모델에서는 S1→S2 전환까지 확인 가능. S2 내부에서 lifted pose 도달은 안 되므로 S2→S3 이후는 수동 테스트 또는 fine-tune 후 검증.

---

## 0. 서버 연결

- [ ] VLM health check 통과 (`VLM: OK` 로그)
- [ ] VLA health check 통과 (`VLA: OK` 로그)
- [ ] VLM /classify가 source/dest 정상 추출 (`[Classify] Result:` 로그에서 source, dest 확인)

## 1. S1 (navigate)

- [ ] VLM이 정확히 50스텝 간격으로 호출되는지 (매 50스텝 로그마다 call_count가 1씩 증가)
- [ ] VLM 응답이 6개 이동 커맨드 또는 TARGET_FOUND 중 하나인지 (`[VLM] Invalid command` 로그가 안 뜨는지)
- [ ] 50스텝 동안 instruction이 변경되지 않는지 (연속된 로그에서 inst= 값 동일)
- [ ] safety layer 동작: depth < 0.3m → base vx, vy 정지, wz 유지 (`[SAFETY]` 로그 출력)
- [ ] VLA에 instruction이 정상 전달되는지 (로그에서 inst= 확인)
- [ ] TARGET_FOUND 시 S2로 전환되는지 (`[SKILL] navigate → approach_and_lift` 로그)
- [ ] TARGET_FOUND가 target이 멀리 있을 때는 출력되지 않고, 가까울 때만 출력되는지

## 2. S1→S2 전환 시점

- [ ] `[SKILL] navigate → approach_and_lift` 로그가 정상 출력
- [ ] VLA action buffer가 reset 되는지 (전환 직후 새 action chunk 요청 발생)
- [ ] instruction이 즉시 `"approach and lift the {source}"` 로 변경되는지
- [ ] `_obstacle_cleared`가 False로 초기화되는지

## 3. S2 (approach & lift)

- [ ] instruction이 S2 내내 고정인지 (로그에서 inst= 값이 바뀌지 않는지)
- [ ] depth warning 없을 때 VLM call_count가 올라가지 않는지
- [ ] depth < 0.3m 발생 시 VLM이 호출되는지 (call_count +1)
- [ ] VLM 응답 CONTINUE → 이후 depth < 0.3m이어도 VLM 재호출 안 되는지 (obstacle_cleared 동작)
- [ ] VLM 응답 OBSTACLE + contact 없음 → S1으로 복귀 (`[SKILL] approach_and_lift → navigate`)
- [ ] VLM 응답 OBSTACLE + contact 있음 → S3로 정상 전환 (`[SKILL] approach_and_lift → carry`)
- [ ] safety layer가 비활성화 상태인지 (S2에서 depth 낮아도 `[SAFETY]` 안 뜨는지)
- [ ] `check_lifted_complete()` 매 스텝 에러 없이 실행되는지
- [ ] timeout 도달 시 `[TIMEOUT]` 로그 정상 출력

## 4. S2→S3 전환 (정상 경로)

base 모델에서는 lifted pose 도달이 안 되므로 실제 발생하지 않을 수 있음.
수동으로 contact=True + lifted pose range 값을 주입하여 테스트하거나, fine-tune 후 검증.

- [ ] `[SKILL] approach_and_lift → carry` 로그
- [ ] instruction이 `"carry forward"` 로 변경
- [ ] VLA buffer reset

## 5. S2에서 OBSTACLE → S1 복귀 → TARGET_FOUND → S2 재진입

- [ ] OBSTACLE 발생 → `[SKILL] approach_and_lift → navigate` 로그
- [ ] S1에서 50스텝 간격 VLM 호출 재개
- [ ] TARGET_FOUND → `[SKILL] navigate → approach_and_lift` 로그
- [ ] 재진입 시 instruction이 다시 `"approach and lift the {source}"` 고정
- [ ] 재진입 시 `_obstacle_cleared`가 다시 False (depth warning 시 VLM 다시 호출됨)

## 6. S3 (carry)

S2→S3 전환이 발생해야 검증 가능.

- [ ] VLM이 50스텝 간격으로 호출되는지
- [ ] 응답이 6개 carry 커맨드 또는 TARGET_FOUND 중 하나인지
- [ ] safety layer 활성화 상태인지 (depth < 0.3m → `[SAFETY]` 로그)
- [ ] TARGET_FOUND 시 S4로 전환 (`[SKILL] carry → approach_and_place`)

## 7. S4 (approach & place)

S3→S4 전환이 발생해야 검증 가능.

- [ ] instruction 고정: `"place the {source} next to the {dest}"`
- [ ] VLM 호출 로직이 S2와 동일 (depth warning + obstacle_cleared)
- [ ] OBSTACLE → S3(carry) 복귀 (`[SKILL] approach_and_place → carry`)
- [ ] `check_place_complete()` 연속 10스텝 카운터 동작
- [ ] DONE 전환 시 `[DONE]` 로그

## 8. 공통

- [ ] 초기 instruction이 `"navigate forward"` 인지
- [ ] S2/S4 프롬프트 format 에러 없음 (prev_instruction 관련 KeyError 없음)
- [ ] `LIFTED_POSE_RANGE` import 에러 없음 (vlm_orchestrator에서만 정의, run_full_task에서 import)
- [ ] 전체 루프가 멈추지 않고 돌아가는지 (Hz 로그 확인)
- [ ] max_total_steps 도달 시 정상 종료
- [ ] Ctrl+C 시 정상 종료

---

## 로그 확인 예시

### S1 정상 동작
```
[t=  50] skill=navigate inst="navigate forward" vlm=191ms(1)
[t= 100] skill=navigate inst="navigate forward" vlm=191ms(2)
[t= 150] skill=navigate inst="navigate turn left" vlm=191ms(3)
```
→ 50스텝마다 call_count +1, instruction이 50스텝간 유지

### S1→S2 전환
```
[SKILL] navigate → approach_and_lift
[t= 350] skill=approach_and_lift inst="approach and lift the medicine bottle" vlm=191ms(7)
[t= 400] skill=approach_and_lift inst="approach and lift the medicine bottle" vlm=191ms(7)
```
→ 전환 후 call_count가 안 올라감 = S2에서 VLM 미호출 정상

### S2에서 depth warning → CONTINUE
```
[t= 450] skill=approach_and_lift inst="approach and lift the medicine bottle" vlm=191ms(8)
[t= 500] skill=approach_and_lift inst="approach and lift the medicine bottle" vlm=191ms(8)
```
→ call_count 8에서 멈춤 = CONTINUE 후 재호출 억제 정상

### S2에서 OBSTACLE → S1 복귀
```
[SKILL] approach_and_lift → navigate
[t= 550] skill=navigate inst="navigate turn right" vlm=191ms(9)
```
→ S1으로 복귀 후 VLM 호출 재개
