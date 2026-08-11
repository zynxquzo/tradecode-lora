# 파인튜닝 학습 로그

## 학습 설정

- base_model: unsloth/gemma-2-2b
- lora_r: 32
- lora_alpha: 64
- target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
- learning_rate: 0.0002
- epochs: 10
- batch_size: 4
- grad_accumulation: 4
- smoke_test: False
- max_steps: N/A (full run)
- train_records: 1408
- val_records: 156

## Loss

| step/epoch | 구분 | loss | elapsed_min |
|---|---|---|---|
| step 10 | train | 0.5939 | 0.7 |
| step 20 | train | 0.2813 | 1.2 |
| step 30 | train | 0.2403 | 1.7 |
| step 40 | train | 0.1474 | 2.2 |
| step 50 | train | 0.1367 | 2.7 |
| step 60 | train | 0.0928 | 3.2 |
| step 70 | train | 0.0992 | 3.7 |
| step 80 | train | 0.0565 | 4.2 |
| step 88 | eval | 0.0952 | 4.7 |
| step 90 | train | 0.0620 | 4.9 |
| step 100 | train | 0.0453 | 5.4 |
| step 110 | train | 0.0398 | 5.9 |
| step 120 | train | 0.0616 | 6.3 |
| step 130 | train | 0.0535 | 6.8 |
| step 140 | train | 0.0430 | 7.3 |
| step 150 | train | 0.0383 | 7.8 |
| step 160 | train | 0.0233 | 8.3 |
| step 170 | train | 0.0251 | 8.9 |
| step 176 | eval | 0.0395 | 9.3 |
| step 180 | train | 0.0241 | 9.5 |
| step 190 | train | 0.0321 | 10.0 |
| step 200 | train | 0.0140 | 10.5 |
| step 210 | train | 0.0179 | 11.0 |
| step 220 | train | 0.0251 | 11.6 |
| step 230 | train | 0.0081 | 12.0 |
| step 240 | train | 0.0142 | 12.5 |
| step 250 | train | 0.0183 | 13.0 |
| step 260 | train | 0.0189 | 13.5 |
| step 264 | eval | 0.0234 | 13.9 |
| step 270 | train | 0.0114 | 14.2 |
| step 280 | train | 0.0094 | 14.7 |
| step 290 | train | 0.0115 | 15.2 |
| step 300 | train | 0.0075 | 15.7 |
| step 310 | train | 0.0063 | 16.2 |
| step 320 | train | 0.0069 | 16.7 |
| step 330 | train | 0.0098 | 17.2 |
| step 340 | train | 0.0094 | 17.7 |
| step 350 | train | 0.0131 | 18.2 |
| step 352 | eval | 0.0090 | 18.4 |
| step 360 | train | 0.0068 | 18.9 |
| step 370 | train | 0.0039 | 19.3 |
| step 380 | train | 0.0083 | 19.8 |
| step 390 | train | 0.0030 | 20.3 |
| step 400 | train | 0.0095 | 20.8 |
| step 410 | train | 0.0038 | 21.3 |
| step 420 | train | 0.0078 | 21.8 |
| step 430 | train | 0.0070 | 22.3 |
| step 440 | train | 0.0105 | 22.8 |
| step 440 | eval | 0.0088 | 23.0 |
| step 450 | train | 0.0104 | 23.5 |
| step 460 | train | 0.0083 | 24.0 |
| step 470 | train | 0.0051 | 24.5 |
| step 480 | train | 0.0044 | 25.0 |
| step 490 | train | 0.0055 | 25.5 |
| step 500 | train | 0.0029 | 26.0 |
| step 510 | train | 0.0003 | 26.5 |
| step 520 | train | 0.0024 | 27.0 |
| step 528 | eval | 0.0013 | 27.6 |
| step 530 | train | 0.0030 | 27.7 |
| step 540 | train | 0.0023 | 28.2 |
| step 550 | train | 0.0010 | 28.7 |
| step 560 | train | 0.0040 | 29.2 |
| step 570 | train | 0.0052 | 29.7 |
| step 580 | train | 0.0024 | 30.1 |
| step 590 | train | 0.0010 | 30.6 |
| step 600 | train | 0.0008 | 31.1 |
| step 610 | train | 0.0014 | 31.6 |
| step 616 | eval | 0.0034 | 32.1 |
