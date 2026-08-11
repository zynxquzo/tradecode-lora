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
- train_records: 1249
- val_records: 139

## Loss

| step/epoch | 구분 | loss | elapsed_min |
|---|---|---|---|
| step 10 | train | 0.5746 | 0.7 |
| step 20 | train | 0.2488 | 1.1 |
| step 30 | train | 0.1379 | 1.7 |
| step 40 | train | 0.1083 | 2.1 |
| step 50 | train | 0.0866 | 2.6 |
| step 60 | train | 0.0998 | 3.1 |
| step 70 | train | 0.0452 | 3.6 |
| step 79 | eval | 0.0431 | 4.2 |
| step 80 | train | 0.0491 | 4.2 |
| step 90 | train | 0.0446 | 4.7 |
| step 100 | train | 0.0552 | 5.2 |
| step 110 | train | 0.0267 | 5.7 |
| step 120 | train | 0.0439 | 6.2 |
| step 130 | train | 0.0205 | 6.7 |
| step 140 | train | 0.0235 | 7.2 |
| step 150 | train | 0.0177 | 7.7 |
| step 158 | eval | 0.0279 | 8.2 |
| step 160 | train | 0.0153 | 8.3 |
| step 170 | train | 0.0086 | 8.8 |
| step 180 | train | 0.0176 | 9.3 |
| step 190 | train | 0.0114 | 9.8 |
| step 200 | train | 0.0196 | 10.3 |
| step 210 | train | 0.0156 | 10.8 |
| step 220 | train | 0.0142 | 11.3 |
| step 230 | train | 0.0146 | 11.8 |
| step 237 | eval | 0.0150 | 12.2 |
| step 240 | train | 0.0066 | 12.4 |
| step 250 | train | 0.0040 | 12.9 |
| step 260 | train | 0.0044 | 13.4 |
| step 270 | train | 0.0034 | 13.8 |
| step 280 | train | 0.0059 | 14.3 |
| step 290 | train | 0.0024 | 14.8 |
| step 300 | train | 0.0074 | 15.3 |
| step 310 | train | 0.0122 | 15.8 |
| step 316 | eval | 0.0054 | 16.2 |
| step 320 | train | 0.0030 | 16.4 |
| step 330 | train | 0.0092 | 16.9 |
| step 340 | train | 0.0025 | 17.4 |
| step 350 | train | 0.0007 | 17.9 |
| step 360 | train | 0.0042 | 18.4 |
| step 370 | train | 0.0018 | 18.9 |
| step 380 | train | 0.0055 | 19.4 |
| step 390 | train | 0.0029 | 19.8 |
| step 395 | eval | 0.0068 | 20.2 |
