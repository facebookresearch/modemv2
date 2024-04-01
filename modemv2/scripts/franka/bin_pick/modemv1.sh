#!/bin/bash

if [ $# -lt 1 ]
then
    MULTI=""
    NAME="exp_name=bin_pick_modemv1"
    SEED="seed=1"
    DEMOS="demos=10"
    LAUNCHER="hydra/launcher=local"
    BATCH_SIZE="batch_size=256"
    SEED_STEPS="seed_steps=5000"
    EVAL_EPISODES="eval_episodes=30"
    EVAL_FREQ="eval_freq=2500"
    MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,0,2500)\""
elif [ $1 = 1 ]
then
    MULTI="-m"
    NAME="exp_name=bin_pick_modemv1_cluster"
    SEED="seed=1,2,3,4,5"
    DEMOS="demos=10"
    LAUNCHER="hydra/launcher=slurm"
    BATCH_SIZE="batch_size=256"
    SEED_STEPS="seed_steps=5000"
    EVAL_EPISODES="eval_episodes=30"
    EVAL_FREQ="eval_freq=2500"
    MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,0,2500)\""
else
    MULTI=""
    NAME="exp_name=bin_pick_modemv1_test"
    SEED="seed=1"
    DEMOS="demos=3"
    LAUNCHER="hydra/launcher=local"    
    BATCH_SIZE="batch_size=16"
    SEED_STEPS="seed_steps=300"
    EVAL_EPISODES="eval_episodes=3"
    EVAL_FREQ="eval_freq=300"
    MIX_SCHEDULE="mix_schedule=\"linear(0.0,1.0,0,100)\""
fi

python train.py  $MULTI \
    task=franka-FrankaBinPick_v2d  \
    $NAME \
    iterations=1\
    discount=0.95 \
    train_steps=200000 \
    $SEED \
    $DEMOS \
    img_size=224 \
    lr=3e-4 \
    $BATCH_SIZE \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[0,0] \
    top_crops=[0,0] \
    action_repeat=1 \
    $SEED_STEPS \
    $EVAL_EPISODES \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=2 \
    val_min_w=1.0 \
    val_mean_w=0.0 \
    val_std_w=0.00 \
    $MIX_SCHEDULE \
    mixture_coef=1.0\
    save_freq=2500\
    $EVAL_FREQ\
    uncertainty_weighting=false\
    vanilla_modem=true\
    min_std=0.1\
    $LAUNCHER
