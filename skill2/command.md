skill2는 약병을 로봇의 전방 0.6~0.9m에 스폰해서 약병을 집을 수 있을 만큼만 이동한 후 그리퍼를 벌리고 팔을 내려 약병쪽으로 팔을 뻗고 그리퍼를 닫아 약병을 grasp 한 후 그대로 팔을 들어올려 최종 lifted pose를 달성하는 것이다.  

train first

cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env && \                                                                                                                                                     
  PYTHONUNBUFFERED=1 LEKIWI_USD_PATH=/home/jovyan/Downloads/lekiwi_robot.usd \
  nohup python train_resip.py \                                                                                                                                                                            
    --skill approach_and_grasp \                                                     
    --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --object_usd /home/jovyan/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
    --num_envs 1024 \
    --lr_actor 1e-3 \
    --target_kl 1.5 \
    --total_timesteps 250000000 \
    --normalize_reward False \
    --headless  
    
이렇게 학습 시켰을때도 이미 약병에 접근해서 팔을 뻗어서 집고 들어올리는거까지는 잘 됨. 단지 reward 안에 lifted pose 설정과 reward scale이 잘못되어 있어 lift 후 팔을 하늘로 뻗는 현상이 있어 train second를 진행


train second

 cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env && \
  PYTHONUNBUFFERED=1 LEKIWI_USD_PATH=/home/jovyan/Downloads/lekiwi_robot.usd \
  nohup python train_resip.py \
    --skill approach_and_grasp \
    --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --object_usd /home/jovyan/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
    --num_envs 512 \
    --lr_actor 1e-4 \
    --lr_critic 1e-3 \
    --update_epochs 10 \
    --num_minibatches 4 \
    --target_kl 0.03 \
    --ent_coef 1e-4 \
    --total_timesteps 250000000 \
    --normalize_reward True \
    --resume_resip /home/jovyan/data/resip_v6.4b_best.pt \
    --resume_actor_only True \
    --warmup_steps_initial 0 \
    --enable_domain_randomization False \
    --r8_penalty -2.0 \
    --save_dir checkpoints/resip_v6.4b_tuned_from66 \
    --headless \
    > logs/resip_v64b_tuned_from66_20260319_025055.log 2>&1 &
    
    
이렇게 해서 lift pose 보완 후 검증시 성공률 100번 중에 64번 성공
