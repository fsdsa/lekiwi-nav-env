✅ 반영 완료: Handoff 좌표계 (AUDIT-V2-1/2)
generate_handoff_buffer.py에서 origin = env.scene.env_origins[i]를 빼고, _reset_from_handoff에서 env_origins를 더하고, home_pos_w를 env_origins[:, 0:2]로 설정하는 것까지 코드 본문에 정확히 반영됐습니다.
단, 설명 문서가 코드와 모순됩니다:

1_전체_파이프라인.md 265행: "Handoff Buffer의 내용(world-frame 절대 좌표 포함)은..."
2_Sim_데이터_수집_파이프라인.md 407행: "base_pos": [x, y, z], # world position
2_Sim_데이터_수집_파이프라인.md 417행: "world-frame 절대 좌표를 포함하지만, sim 내부 reset 용도이므로 괜찮다."

코드는 이미 env_origin 상대 좌표로 바뀌었으므로 이 설명들도 맞춰야 합니다.

❌ 미반영: _action_delay_buf 리셋 (AUDIT-V2-3)
3_코드_현황_정리.md 919행의 _finish_reset 코드 본문:
pythondef _finish_reset(self, env_ids, num):
    self.task_success[env_ids] = False
    ...
    self.actions[env_ids] = 0.0
    self.prev_actions[env_ids] = 0.0
    # ← _action_delay_buf 리셋이 없음
추적표에 "완료"로 되어 있지만 코드 본문에 if self._action_delay_buf is not None: self._action_delay_buf[:, env_ids] = 0.0이 없습니다.

❌ 미반영: Curriculum window 클리어 (AUDIT-V2-5)
codefix.md 750행과 3_코드_현황_정리.md 430행의 curriculum 코드:
pythonif self._curriculum_dist != old:
    print(f"  [Curriculum] dist: {old:.2f} -> ...")
    # ← window.zero_() + idx = 0 이 없음
마찬가지로 추적표에 "완료"지만 코드 본문에 반영 안 됐습니다.

✅ 비적용 항목 처리 적절
prev_object_dist(V2-5b)와 clip_predicted_values(V2-6)를 취소선 + "비적용"으로 처리한 건 맞습니다. 다만 1_전체_파이프라인.md 257행에는 아직 6개 항목이 모두 나열되어 있어서 취소선 처리와 불일치합니다.
