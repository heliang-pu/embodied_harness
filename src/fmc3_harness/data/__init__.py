"""Turns lerobot-annotate datasets into what the harness and its models need.

- labels: map free-form subtask labels onto the skill vocabulary (reviewed by a human);
- library: fill prompt-library statistics (episodes, durations) from the annotations;
- planner: build fine-tuning samples for the planner with the shared renderer;
- relabel: copy a dataset with canonical subtask texts, for training the executor.
"""
