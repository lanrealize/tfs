import tensorflow as tf

from model import TFSModel, train_step, model_evaluate
from data import build_ds

# Train parameters
EPOCH = 1000
EVALUATE_INTERVAL = 7
LEARNING_RATE = 0.005

train_ds, val_ds = build_ds()

model = TFSModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.CategoricalCrossentropy()
loss_metrics = tf.keras.metrics.Mean()

for epoch in range(EPOCH):
    print(f'epoch {epoch}:')
    batch_idx = 0
    for inputs in train_ds:
        model_evaluate(model, val_ds)
        train_step(model=model,
                   optimizer=optimizer,
                   loss=loss,
                   loss_metrics=loss_metrics,
                   batched_inputs=inputs['inputs'],
                   batched_targets=inputs['targets'])

        if batch_idx % EVALUATE_INTERVAL == 0:
            pass
        batch_idx += 1
    print(f"train loss is: {loss_metrics.result().numpy()}")

    loss_metrics.reset_states()
