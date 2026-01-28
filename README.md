# PromptForest - Fast and Reliable Injection Detector Ensemble

PromptForest is a prompt injection detector ensemble focused on real-world latency and reliability.

We rely on an ensemble of small, accurate prompt detection models using a voting system to generate accurate detections. 

By comparing predictions across multiple models, the system can flag prompts where models disagree, helping to reduce the risk of false negatives.

This discrepancy score enables downstream workflows such as:
- Human-in-the-loop review for high-risk or ambiguous prompts  
- Adaptive throttling or alerting in production systems  
- Continuous monitoring and model improvement  


## Supported Models

| Provider      | Model Name                 |
| ------------- | ----------------------------------------- |
| **Meta**      | Llama Prompt Guard 86M (Built with Llama) |
| **ProtectAI** | DebertaV3 Prompt Injection Finetune       |
| **Vijil**     | Vijil Dome Prompt Injection Detection     |
| **Appleroll** | PromptForest-XGBoost                      |

## Performance
**Request Latency** \
Best Case: 50ms \
Worst Case: 200ms

**Accuracy** \
Preliminary results indicate ensemble performance is at least as good as any individual model. Extensive benchmarking is ongoing.


## Quick Start
To use PromptForest, simply install the pip package and serve it at a port of your choice. It should automatically start downloading the default model ensemble.

Gated models are downloaded through our own [ensemble github respository](https://github.com/appleroll-research/promptforest-model-ensemble) and are released in accordance to their terms and conditions.

```bash
pip install promptforest
promptforest serve --port 8000 
```

## Disclaimer & Limitations

PromptForest uses a combination of open-source and third-party machine learning models, including models and weights released by other organizations under their respective licenses (e.g. Meta LLaMA Prompt Guard and other public prompt-injection detectors).
All third-party components remain the property of their original authors and are used in accordance with their licenses.

PromptForest is not a standalone security solution and should not be relied upon as the sole defense mechanism for protecting production systems. Prompt injection detection is an inherently adversarial and evolving problem, and no automated system can guarantee complete protection.

This project has not yet been extensively validated against real-world, large-scale, or targeted prompt-injection attacks. Results may vary depending on deployment context, model configuration, and threat model.

PromptForest is intended to be used as one layer in a defense-in-depth strategy, alongside input validation, output filtering, access control, sandboxing, monitoring, and human oversight.

## License
This project is licensed under Apache 2.0. Third-party models and weights are redistributed under their original licenses (see THIRD_PARTY_LICENSES folder for details). Users must comply with these licenses.
