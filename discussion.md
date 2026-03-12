Exp 01
Training Results — SauberModel3 (50 Epochs)
Milestone	Epoch	Loss	Accuracy	Recall
Start	1	0.4453	80.6%	80.4%
Converging	10	0.0674	97.8%	98.7%
Peak	40	0.0040	100.0%	100.0%
Final	50	0.0180	99.4%	99.7%
Key observations:

Epoch 1 took ~11 min — that's the one-time cuDNN benchmarking overhead picking optimal kernels
Epochs 2–50 settled at ~3.4 min — AMP fp16 + persistent workers working as expected
Total training: ~2h55min for 50 epochs on the RTX 4070 Ti SUPER
Slight overfitting at epoch 40 (100% train acc), then a minor drop-off — normal for this dataset size; the final 99.4%/99.7% is still excellent
The model achieves very strong recall (99.7%) which is the most clinically critical metric — very few actual pneumonia cases are missed. 

Exp 02
Here's a clean summary of the final run:

Phase	Epochs	Accuracy	Recall
Early convergence	1–5	76% → 96%	71% → 98%
Steady improvement	6–30	96% → 99%	98% → 99.7%
Near-perfect	31–47	99.7% → 100%	99.8% → 100%
Final (epoch 50)	50	99.8%	99.8%
Key observations from this run:

Epoch 1 was ~14.7 min — cuDNN benchmarking overhead (one-time cost, expected)
Epochs 2–50 stabilised at ~3.5 min each — AMP + 4 workers in full effect
Total training: ~3h10min for 50 epochs in Jupyter (slightly slower than the 

.py
 run due to 4 vs 8 workers — the safe tradeoff for Jupyter stability on Windows)
The model reached 100% train accuracy at epoch 39 and held it through epochs 44–48