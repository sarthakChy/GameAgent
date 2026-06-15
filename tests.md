!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/vjepa2_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --sequence-length 8 \
  --temporal-hidden-dim 256 \
  --repeat-decay 1.0 \
  --inverse-dynamics-weight 0.25 \
  --batch-size 32 \
  --epochs 15 \
  --device cuda \
  --use-amp

  Loaded bucketized vocab: data_processing/outputs/action_vocab.json
/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:524: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260430_083913
  train_samples: 46058
  val_samples: 9215
  vocab_size: 120
  vision_dim: 1408
  sequence_length: 8
  temporal_hidden_dim: 256
  inverse_dynamics_classes: 30061
  device: cuda
  amp: True
epoch=1/15 train_loss=2.2486 val_loss=0.3407 lr=2.967403e-04
epoch=2/15 train_loss=1.6294 val_loss=0.3262 lr=2.870673e-04
epoch=3/15 train_loss=1.1915 val_loss=0.3209 lr=2.714166e-04
epoch=4/15 train_loss=0.8526 val_loss=0.3272 lr=2.504668e-04
epoch=5/15 train_loss=0.6431 val_loss=0.3174 lr=2.251133e-04
epoch=6/15 train_loss=0.5284 val_loss=0.3201 lr=1.964978e-04
epoch=7/15 train_loss=0.4585 val_loss=0.3135 lr=1.658311e-04
epoch=8/15 train_loss=0.4121 val_loss=0.3185 lr=1.344943e-04
epoch=9/15 train_loss=0.3784 val_loss=0.3118 lr=1.038342e-04
epoch=10/15 train_loss=0.3532 val_loss=0.3157 lr=7.517011e-05
epoch=11/15 train_loss=0.3333 val_loss=0.3178 lr=4.977641e-05
epoch=12/15 train_loss=0.3183 val_loss=0.3169 lr=2.877581e-05
epoch=13/15 train_loss=0.3083 val_loss=0.3166 lr=1.306597e-05
epoch=14/15 train_loss=0.3019 val_loss=0.3186 lr=3.327943e-06
epoch=15/15 train_loss=0.2992 val_loss=0.3193 lr=2.284630e-10
Training complete.
  best_val_loss: 0.3118
  run_dir: data_processing/outputs/action_decoder_runs/20260430_083913
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260430_083913/best.pt




!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/vjepa2_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --sequence-length 8 \
  --temporal-hidden-dim 0 \
  --repeat-decay 1.0 \
  --inverse-dynamics-weight 0.0 \
  --batch-size 32 \
  --epochs 15 \
  --device cuda \
  --use-amp

Loaded bucketized vocab: data_processing/outputs/action_vocab.json
/teamspace/studios/this_studio/GameAgent/data_processing/action_model.py:81: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:524: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260503_080626
  train_samples: 46058
  val_samples: 9215
  vocab_size: 120
  vision_dim: 1408
  sequence_length: 8
  temporal_hidden_dim: 0
  inverse_dynamics_classes: 30061
  device: cuda
  amp: True
epoch=1/15 train_loss=0.3930 val_loss=0.3530 lr=2.967221e-04
epoch=2/15 train_loss=0.3175 val_loss=0.3247 lr=2.870318e-04
epoch=3/15 train_loss=0.3025 val_loss=0.3271 lr=2.713525e-04
epoch=4/15 train_loss=0.2942 val_loss=0.3203 lr=2.503696e-04
epoch=5/15 train_loss=0.2871 val_loss=0.3185 lr=2.250000e-04
epoch=6/15 train_loss=0.2805 val_loss=0.3135 lr=1.963733e-04
epoch=7/15 train_loss=0.2750 val_loss=0.3129 lr=1.657227e-04
epoch=8/15 train_loss=0.2691 val_loss=0.3112 lr=1.343641e-04
epoch=9/15 train_loss=0.2636 val_loss=0.3092 lr=1.037097e-04
epoch=10/15 train_loss=0.2581 val_loss=0.3089 lr=7.507559e-05
epoch=11/15 train_loss=0.2528 val_loss=0.3115 lr=4.969528e-05
epoch=12/15 train_loss=0.2491 val_loss=0.3096 lr=2.871160e-05
epoch=13/15 train_loss=0.2454 val_loss=0.3092 lr=1.302148e-05
epoch=14/15 train_loss=0.2435 val_loss=0.3098 lr=3.309687e-06
epoch=15/15 train_loss=0.2420 val_loss=0.3100 lr=7.774090e-11
Training complete.
  best_val_loss: 0.3089
  run_dir: data_processing/outputs/action_decoder_runs/20260503_080626
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260503_080626/best.pt







!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/vjepa2_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --sequence-length 8 \
  --temporal-hidden-dim 0 \
  --repeat-decay 1.0 \
  --inverse-dynamics-weight 0.25 \
  --batch-size 32 \
  --epochs 15 \
  --device cuda \
  --use-amp

Loaded bucketized vocab: data_processing/outputs/action_vocab.json
/teamspace/studios/this_studio/GameAgent/data_processing/action_model.py:81: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:524: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260503_081201
  train_samples: 46058
  val_samples: 9215
  vocab_size: 120
  vision_dim: 1408
  sequence_length: 8
  temporal_hidden_dim: 0
  inverse_dynamics_classes: 30061
  device: cuda
  amp: True
epoch=1/15 train_loss=2.2531 val_loss=0.3536 lr=2.967403e-04
epoch=2/15 train_loss=1.6510 val_loss=0.3242 lr=2.870673e-04
epoch=3/15 train_loss=1.2230 val_loss=0.3255 lr=2.714166e-04
epoch=4/15 train_loss=0.8923 val_loss=0.3277 lr=2.504668e-04
epoch=5/15 train_loss=0.6810 val_loss=0.3207 lr=2.251133e-04
epoch=6/15 train_loss=0.5602 val_loss=0.3155 lr=1.964978e-04
epoch=7/15 train_loss=0.4884 val_loss=0.3192 lr=1.658528e-04
epoch=8/15 train_loss=0.4399 val_loss=0.3121 lr=1.344943e-04
epoch=9/15 train_loss=0.4057 val_loss=0.3114 lr=1.038342e-04
epoch=10/15 train_loss=0.3797 val_loss=0.3087 lr=7.518902e-05
epoch=11/15 train_loss=0.3597 val_loss=0.3105 lr=4.979264e-05
epoch=12/15 train_loss=0.3451 val_loss=0.3088 lr=2.877581e-05
epoch=13/15 train_loss=0.3347 val_loss=0.3098 lr=1.306597e-05
epoch=14/15 train_loss=0.3285 val_loss=0.3103 lr=3.332514e-06
epoch=15/15 train_loss=0.3259 val_loss=0.3105 lr=2.681267e-10
Training complete.
  best_val_loss: 0.3087
  run_dir: data_processing/outputs/action_decoder_runs/20260503_081201
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260503_081201/best.pt








# simplified architecture


!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/vjepa2_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --batch-size 32 \
  --epochs 15 \
  --device cuda \
  --use-amp


Built vocab and saved: data_processing/outputs/action_vocab.json
/teamspace/studios/this_studio/GameAgent/data_processing/action_model.py:59: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:310: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260503_084433
  train_samples: 52576
  val_samples: 2767
  vocab_size: 120
  vision_dim: 1408
  device: cuda
  amp: True
epoch=1/15 train_loss=0.3915 val_loss=0.3218 lr=2.967221e-04
epoch=2/15 train_loss=0.3175 val_loss=0.3032 lr=2.870318e-04
epoch=3/15 train_loss=0.3043 val_loss=0.2974 lr=2.713525e-04
epoch=4/15 train_loss=0.2958 val_loss=0.2910 lr=2.503696e-04
epoch=5/15 train_loss=0.2887 val_loss=0.2841 lr=2.250166e-04
epoch=6/15 train_loss=0.2826 val_loss=0.2809 lr=1.963707e-04
epoch=7/15 train_loss=0.2770 val_loss=0.2789 lr=1.657173e-04
epoch=8/15 train_loss=0.2711 val_loss=0.2733 lr=1.343778e-04
epoch=9/15 train_loss=0.2662 val_loss=0.2699 lr=1.037202e-04
epoch=10/15 train_loss=0.2608 val_loss=0.2680 lr=7.508281e-05
epoch=11/15 train_loss=0.2560 val_loss=0.2657 lr=4.971570e-05
epoch=12/15 train_loss=0.2519 val_loss=0.2634 lr=2.871492e-05
epoch=13/15 train_loss=0.2486 val_loss=0.2628 lr=1.302268e-05
epoch=14/15 train_loss=0.2468 val_loss=0.2623 lr=3.309740e-06
epoch=15/15 train_loss=0.2455 val_loss=0.2623 lr=9.871618e-11
Training complete.
  best_val_loss: 0.2623
  run_dir: data_processing/outputs/action_decoder_runs/20260503_084433
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260503_084433/best.pt



!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/vjepa2_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --batch-size 64 \
  --epochs 15 \
  --device cuda \
  --use-amp


Loaded bucketized vocab: data_processing/outputs/action_vocab.json
/teamspace/studios/this_studio/GameAgent/data_processing/action_model.py:59: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:310: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260503_084813
  train_samples: 52576
  val_samples: 2767
  vocab_size: 120
  vision_dim: 1408
  device: cuda
  amp: True
epoch=1/15 train_loss=0.4199 val_loss=0.3244 lr=2.967221e-04
epoch=2/15 train_loss=0.3190 val_loss=0.3041 lr=2.870318e-04
epoch=3/15 train_loss=0.3047 val_loss=0.3008 lr=2.713525e-04
epoch=4/15 train_loss=0.2962 val_loss=0.2914 lr=2.503696e-04
epoch=5/15 train_loss=0.2894 val_loss=0.2856 lr=2.250000e-04
epoch=6/15 train_loss=0.2837 val_loss=0.2794 lr=1.963525e-04
epoch=7/15 train_loss=0.2781 val_loss=0.2796 lr=1.656793e-04
epoch=8/15 train_loss=0.2728 val_loss=0.2756 lr=1.343207e-04
epoch=9/15 train_loss=0.2678 val_loss=0.2722 lr=1.036475e-04
epoch=10/15 train_loss=0.2632 val_loss=0.2706 lr=7.503310e-05
epoch=11/15 train_loss=0.2587 val_loss=0.2683 lr=4.968723e-05
epoch=12/15 train_loss=0.2549 val_loss=0.2661 lr=2.869240e-05
epoch=13/15 train_loss=0.2522 val_loss=0.2657 lr=1.299929e-05
epoch=14/15 train_loss=0.2504 val_loss=0.2650 lr=3.293771e-06
epoch=15/15 train_loss=0.2495 val_loss=0.2650 lr=1.947578e-11
Training complete.
  best_val_loss: 0.2650
  run_dir: data_processing/outputs/action_decoder_runs/20260503_084813
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260503_084813/best.pt



# after correcting the train val split nd data leakage
!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/train_embeddings.index.json \
  --val-index-path data_processing/outputs/val_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --batch-size 32 \
  --epochs 15 \
  --device cuda \
  --use-amp

Loaded bucketized vocab: data_processing/outputs/action_vocab.json
/teamspace/studios/this_studio/GameAgent/data_processing/action_model.py:59: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:310: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260503_114359
  train_samples: 40718
  val_samples: 14625
  vocab_size: 120
  vision_dim: 1408
  device: cuda
  amp: True

epoch=1/15 train_loss=0.4010 val_loss=1.8544 lr=2.967221e-04
epoch=2/15 train_loss=0.3240 val_loss=1.7398 lr=2.870318e-04
epoch=3/15 train_loss=0.3103 val_loss=1.8486 lr=2.713525e-04
epoch=4/15 train_loss=0.3010 val_loss=1.9564 lr=2.503696e-04
epoch=5/15 train_loss=0.2947 val_loss=2.0650 lr=2.250000e-04
epoch=6/15 train_loss=0.2882 val_loss=2.0259 lr=1.963525e-04
epoch=7/15 train_loss=0.2820 val_loss=2.1808 lr=1.657038e-04
epoch=8/15 train_loss=0.2769 val_loss=2.2077 lr=1.343698e-04
epoch=9/15 train_loss=0.2711 val_loss=2.2255 lr=1.036944e-04
epoch=10/15 train_loss=0.2657 val_loss=2.2528 lr=7.506413e-05
epoch=11/15 train_loss=0.2610 val_loss=2.2685 lr=4.968544e-05
epoch=12/15 train_loss=0.2568 val_loss=2.3060 lr=2.870550e-05
epoch=13/15 train_loss=0.2531 val_loss=2.3081 lr=1.301842e-05
epoch=14/15 train_loss=0.2510 val_loss=2.3265 lr=3.303564e-06
epoch=15/15 train_loss=0.2501 val_loss=2.3243 lr=7.308431e-11
Training complete.
  best_val_loss: 1.7398
  run_dir: data_processing/outputs/action_decoder_runs/20260503_114359
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260503_114359/best.pt


!PYTHONPATH=. python data_processing/train_action_decoder.py \
  --index-path data_processing/outputs/train_embeddings.index.json \
  --val-index-path data_processing/outputs/val_embeddings.index.json \
  --vocab-path data_processing/outputs/action_vocab.json \
  --batch-size 64 --epochs 15 --device cuda --use-amp

Loaded bucketized vocab: data_processing/outputs/action_vocab.json
/teamspace/studios/this_studio/GameAgent/data_processing/action_model.py:59: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
/teamspace/studios/this_studio/GameAgent/data_processing/train_action_decoder.py:310: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
Training config:
  run_dir: data_processing/outputs/action_decoder_runs/20260503_114714
  train_samples: 40718
  val_samples: 14625
  vocab_size: 120
  vision_dim: 1408
  device: cuda
  amp: True
epoch=1/15 train_loss=0.4343 val_loss=1.7450 lr=2.967221e-04
epoch=2/15 train_loss=0.3254 val_loss=1.7364 lr=2.870318e-04
epoch=3/15 train_loss=0.3109 val_loss=1.8458 lr=2.713525e-04
epoch=4/15 train_loss=0.3017 val_loss=1.9322 lr=2.503696e-04
epoch=5/15 train_loss=0.2951 val_loss=1.9049 lr=2.250000e-04
epoch=6/15 train_loss=0.2893 val_loss=1.9337 lr=1.963525e-04
epoch=7/15 train_loss=0.2832 val_loss=1.9571 lr=1.656793e-04
epoch=8/15 train_loss=0.2778 val_loss=2.0122 lr=1.343207e-04
epoch=9/15 train_loss=0.2729 val_loss=1.9867 lr=1.036475e-04
epoch=10/15 train_loss=0.2682 val_loss=2.0233 lr=7.500000e-05
epoch=11/15 train_loss=0.2637 val_loss=1.9811 lr=4.963041e-05
epoch=12/15 train_loss=0.2603 val_loss=2.0004 lr=2.864745e-05
epoch=13/15 train_loss=0.2574 val_loss=2.0398 lr=1.298825e-05
epoch=14/15 train_loss=0.2551 val_loss=2.0285 lr=3.288122e-06
epoch=15/15 train_loss=0.2541 val_loss=2.0296 lr=3.243094e-11
Training complete.
  best_val_loss: 1.7364
  run_dir: data_processing/outputs/action_decoder_runs/20260503_114714
  best_checkpoint: data_processing/outputs/action_decoder_runs/20260503_114714/best.pt