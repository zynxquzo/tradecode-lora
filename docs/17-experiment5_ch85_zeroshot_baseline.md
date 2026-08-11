# Evaluation Result

- Model: `gemma2:2b`
- Prompt style: `zero_shot`
- Code length: 4
- Samples: 220
- Generated at: 2026-08-11T23:00:09+09:00

## Metrics

| Metric | Score |
|---|---|
| Exact Match (4-digit) | 2.73% |
| Partial Match (2-digit) | 33.18% |
| Top-3 Recall | 5.91% |
| Parse Failure Rate | 4.09% |

## Sample Predictions

| Input | True | Predicted | Exact | Top-3 |
|---|---|---|---|---|
| Semiconductor devices (for example, diodes, transi | 8541 | 84711900, 84711100, 84719900 | ❌ | ❌ |
| Semiconductor devices like diodes and transistors, | 8541 | 847110, 847990, 845190 | ❌ | ❌ |
| Electronic components including semiconductor devi | 8541 | 8471, 8471, 8471 | ❌ | ❌ |
| Includes semiconductor devices such as diodes and  | 8541 | 8471 | ❌ | ❌ |
| Thermionic, cold cathode or photo-cathode valves a | 8540 | 8471, 8471, 8472 | ❌ | ❌ |
| Thermionic, cold cathode, or photo-cathode valves  | 8540 | 847110, 847120, 847190 | ❌ | ❌ |
| Includes thermionic, cold cathode, and photo-catho | 8540 | 847199, 850190, 847132 | ❌ | ❌ |
| Valves and tubes like thermionic, cold cathode, an | 8540 | 8471, 8471, 8471 | ❌ | ❌ |
| Electrical and electronic waste and scrap - Contai | 8549 | 8471, 8471, 8471 | ❌ | ❌ |
| Mixed electrical and electronic waste including ba | 8549 | 847190, 847190, 846710 | ❌ | ❌ |
| Assorted e-waste and scrap with batteries, electri | 8549 | 8471, 8471, 8472 | ❌ | ❌ |
| Electrical and electronic scrap including primary  | 8549 | 847190, 850410, 740190 | ❌ | ❌ |
| Insulated (including enamelled or anodised) wire,  | 8544 | 85369900, 853610, 853620 | ❌ | ❌ |
| Insulated wire and cable, including co-axial types | 8544 | 8541, 8542, 8542 | ❌ | ❌ |
| Wire and cables that are insulated, like co-axial  | 8544 | 85369900, 85361000, 85471000 | ❌ | ❌ |
| Various insulated electric cables and wires, inclu | 8544 | 8536, 8536, 8536 | ❌ | ❌ |
| Electrical capacitors, fixed, variable or adjustab | 8532 | 8509, 8509, 8509 | ❌ | ❌ |
| Fixed, variable, or adjustable electrical capacito | 8532 | 81159000, 84719000, 85362000 | ❌ | ❌ |
| Parts for fixed, variable, or adjustable electrica | 8532 | 850611, 850612, 850619 | ❌ | ❌ |
| Electrical capacitor parts, fixed or variable, inc | 8532 | 8506, 8506 | ❌ | ❌ |
