# PromptForest

PromptForest is a prompt injection detector ensemble focused on real-world latency and reliability.

## Quick Start
To use PromptForest, simply install the pip package and serve it at a port of your choice. It should automatically start downloading the default model ensemble.

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

Use at your own risk.

## License
This project is under the Apache license.