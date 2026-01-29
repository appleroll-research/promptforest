# PromptForest - Fast and Reliable Injection Detector Ensemble
![PyPI Downloads](https://img.shields.io/pypi/dm/promptforest)
![Apache License](https://img.shields.io/badge/license-Apache%20License%202.0-blue)

**📖 TRY IT OUT ON A NOTEBOOK [HERE](https://colab.research.google.com/drive/1EW49Qx1ZlaAYchqplDIVk2FJVzCqOs6B?usp=sharing)!**

PromptForest is a prompt injection detector ensemble focused on real-world latency and reliability.

We rely on an ensemble of small, accurate prompt detection models using a voting system to generate accurate detections. 

By comparing predictions across multiple models, the system can flag prompts where models disagree, helping to reduce the risk of false negatives.

This discrepancy score enables downstream workflows such as:
- Human-in-the-loop review for high-risk or ambiguous prompts  
- Adaptive throttling or alerting in production systems  
- Continuous monitoring and model improvement  

## Quick Start
To use PromptForest, simply install the pip package and serve it at a port of your choice. It should automatically start downloading the default model ensemble.

Gated models are downloaded through our own [ensemble github respository](https://github.com/appleroll-research/promptforest-model-ensemble) and are released in accordance to their terms and conditions.

```bash
pip install promptforest
promptforest serve --port 8000 
```

## Statistics
**E2E Request Latency** \
Average Case: 100ms \
Worst Case: 200ms

PromptForest was evaluated against the models from Deepset, ProtectAI, Meta and Vijil, with Promptforest and the SOTA model Qualifire's Sentinel V2 performing the best in terms of reliability.

| Metric                           | PromptForest | Sentinel v2 |
| -------------------------------- | ------------ | ----------- |
| Accuracy                         | 0.802        | 0.982       |
| Avg Confidence on Wrong Answers  | 0.659        | 0.782       |
| Expected Calibration Error (ECE) | 0.049        | 0.060       |
| Approximate Model Size           | ~300M params  | ~600M params |


### Key Insights

- Calibrated uncertainty: PromptForest is less confident on wrong predictions than Sentinel, resulting in a much lower ECE.

- Parameter efficiency: Achieves competitive reliability with <50% of the parameters.

- Interpretability: Confidence scores can be used to flag uncertain predictions for human review.

Interpretation:
While Sentinel has higher raw accuracy, PromptForest provides better-calibrated confidence. For systems where overconfidence on wrong answers is risky, PromptForest can reduce the chance of critical errors despite being smaller and faster. 

Using Sentinel v2 as a baseline, and given that other models perform worse than Sentinel in published benchmarks, PromptForest is expected to offer more reliable and calibrated predictions than most alternatives.


## Supported Models

| Provider      | Model Name                 |
| ------------- | ----------------------------------------- |
| **Meta**      | [Llama Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M) (Built with Llama) |
| **ProtectAI** | [DebertaV3 Prompt Injection Finetune](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)       |
| **Vijil**     | [Vijil Dome Prompt Injection Detection](https://huggingface.co/vijil/vijil_dome_prompt_injection_detection)     |
| **Appleroll** | [PromptForest-XGB](https://huggingface.co/appleroll/promptforest-xgb)                      |

## Current Goals
This project is actively being updated. Our current focus is on implementing weights on individual models to improve accuracy, as well as retraining the XGBoost model with an updated corpus.

## Disclaimer & Limitations

PromptForest uses a combination of open-source and third-party machine learning models, including models and weights released by other organizations under their respective licenses (e.g. Meta LLaMA Prompt Guard and other public prompt-injection detectors).
All third-party components remain the property of their original authors and are used in accordance with their licenses.

PromptForest is not a standalone security solution and should not be relied upon as the sole defense mechanism for protecting production systems. Prompt injection detection is an inherently adversarial and evolving problem, and no automated system can guarantee complete protection.

This project has not yet been extensively validated against real-world, large-scale, or targeted prompt-injection attacks. Results may vary depending on deployment context, model configuration, and threat model.

PromptForest is intended to be used as one layer in a defense-in-depth strategy, alongside input validation, output filtering, access control, sandboxing, monitoring, and human oversight.

## License
This project is licensed under Apache 2.0. Third-party models and weights are redistributed under their original licenses (see THIRD_PARTY_LICENSES folder for details). Users must comply with these licenses.
