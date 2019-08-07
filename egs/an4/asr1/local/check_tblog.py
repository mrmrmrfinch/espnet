# coding: utf-8
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator("./tensorboard/train_nodev_pytorch_train_mtlalpha1.0")
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())
try:
    t = event_acc.Scalars("main/cer_ctc")
except KeyError:
    t = {}

try:
    v = event_acc.Scalars("validation/main/cer_ctc")
except KeyError:
    v = {}
# print(v)

print("-----")
print(f"#train: {len(t)}, #valid {len(v)}")
