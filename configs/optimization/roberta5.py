from hyperopt import hp


space = {
    'train_params.num_epochs': hp.quniform('train_params.num_epochs', 3, 8, 1),
    'optimizer.lr': hp.uniform('optimizer.lr', 0.000015, 0.00005),
    'scheduler.frac_training_steps': hp.uniform('scheduler.frac_training_steps', 0, 0.15),
    'dataloader.batch_size': hp.choice('dataloader.batch_size', [16, 32]),
    'criterion.heads_reduction': hp.choice('criterion.heads_reduction', ["sum", "mean"]),
    'criterion.__class__': hp.choice('criterion.__class__', [
        "losses.QACrossEntropyLoss",
        "losses.SoftCrossEntropyLoss",
        "losses.SmoothCrossEntropyLoss",
    ]),
    'model.num_take_layers': hp.quniform('model.num_take_layers', .5, 10.5, 1),
    'model.layers_agg': hp.choice('model.layer_agg', ["concat", "sum"]),
    'model.multi_sample_dropout':  hp.choice('model.multi_sample_dropout', [True, False]),
    'model.dropout': hp.uniform('model.dropout', 0.05, 0.15),
    'model.pre_head_dropout': hp.uniform('model.pre_head_dropout', 0.05, 0.15),
    'model.p_drophead': hp.choice('model.p_drophead', [
        None,
        hp.uniform('model.p_drophead_val', 0.02, 0.2)
    ])
}
