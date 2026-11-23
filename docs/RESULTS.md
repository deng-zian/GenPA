# Results and evaluation protocols

The paper reports success rates for standard LIBERO, LIBERO-Plus and RoboTwin 2.0. [Paper on arXiv](https://arxiv.org/abs/2611.27446)

## Main checkpoint

The main LIBERO depth policy uses training seed 42 and checkpoint step 32,000. It predicts 16 actions and replans after 12 actions, with a 512-wide action adapter. All three scores below use the same checkpoint.

| Protocol | Successes / trials | Success rate |
|---|---:|---:|
| Standard LIBERO, original selection states | 1,982 / 2,000 | 99.10% |
| LIBERO-Plus, Clean instructions | 8,404 / 10,030 | 83.79% |
| LIBERO-Plus, Official instructions | 8,326 / 10,030 | 83.01% |

**99.10% uses the checkpoint-selection states.** Appendix C.1 defines selection by standard LIBERO successes with the earlier step winning ties. The released seed-42 depth checkpoint is the unique standard-scan maximum in Table 24. The same weights are used on Plus. The independent-state result is given below.

Standard LIBERO: four suites × ten tasks × 50 trials; environment seed 42; original released initial states; 30 no-op warm-up steps; action-step limits of 400 for Spatial/Object/Goal and 700 for Long; two rendered 256×256 cameras resized by the policy to 224×224 each. Evaluation uses BF16 and no action ensembling.

Plus: all 10,030 task instances, one trial per instance, environment seed 42 and the first released initial state. Overall success is **pooled across instances**, not an unweighted mean of suite or category percentages. Clean removes exact appended perturbation identifiers in 8,493 non-language instances and preserves all 1,537 Language-category instructions; Official uses released instructions unchanged. Separate benchmark configs, prompt caches and output directories are required.

## Repeated training and independent-state evaluation

| Depth policy experiment | Standard LIBERO | Plus Clean | Plus Official |
|---|---:|---:|---:|
| Three training seeds, selected checkpoints | 98.87 ± 0.25 | 82.33 ± 1.27 | 78.67 ± 3.97 |

Values are percent, mean ± sample standard deviation across training seeds 42, 142 and 242. Standard scores reuse selection states.

The fixed seed-42 depth policy additionally scores **98.33 ± 0.29%** on three shared independent initial-state sets (6,000 episodes in total). Here variation is across evaluation sets, not training runs. The standard evaluation command uses the original released initial states.

## RoboTwin 2.0

The released 64,000-step checkpoint scored 92.08% on Clean and 90.82% on Randomized (91.45% macro-average). Each condition uses 50 tasks × 100 trials. See Table 3 and Appendix C.4.

## LIBERO re-evaluation

Using the same 32,000-step depth checkpoint on eight H100s, a second evaluation achieved 98.75% on standard LIBERO (1,975/2,000) and 83.84% on LIBERO-Plus Clean (8,409/10,030). It used the original selection states for standard LIBERO and the first released state for each Plus instance. [Worker settings and runtime](EVALUATION.md#measured-h100-configurations)
