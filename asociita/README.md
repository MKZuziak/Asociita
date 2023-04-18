### Setting Configuration

In order to run the simulation, the `Orchestrator` instance must receive a `settings` object that contains all the necessary parameters. It is possible to store those parameters in a `JSON` format and load them as the Python dictionary by using `asociita.utils.helper.load_from_json` function. Below is an exemplary settings object embedded as a `json` file. All the elements are necessary unless stated otherwise.

```
{
    "orchestrator":{
        "iterations": int,
        "number_of_nodes": int,
        "local_warm_start": bool,
        "sample_size": int,
        "evaluation": "none" | "full"
        "save_metrics": bool,
	"save_models": bool,
	"save_path": str
	"nodes": [0,
	1,
	2]
    },
    "nodes":{
    "local_epochs": int,
    "model_settings": {
        "optimizer": "RMS",
        "batch_size": int,
        "learning_rate": float}
        }
}
```

The `settings` contains two dictionaries: `orchestrator` and `nodes`.

`orchestrator` contains all the settings necessary details of the training:

* `iterations` is the number of rounds to be performed. Example: `iterations:12`
* `number_of_nodes` is the number of nodes that will be included in the training. Example: `number_of_nodes: 10`
* `local_warm_start` allows to distribute various pre-trained weights to different local clients. Not implemeneted yet. Example: `local_warm_start: false`.
* `sample_size` is the size of the sample that will be taken each round. Example: `sample_size : 4.`
* `evaluation` allows to control the evaluation procedure across the clients.  Currently, only `none` or `full` are supported. Setting the evaluation to full will perform a full evaluation of every client included in the training. Example: `evaluation: full`
* `save_metrics` allows to control whether the metrics should be saved in a csv file. Example: `save_metrics: true.`
* `save_models` allows to control whether the models should be saved. Not implemeneted yet. Example: `save_metrics: false`.
* `save_path` is the system path that will be used when saving the model. It is possible to define a saving_path in a method call.
* `nodes` is the list containing the ids of all the nodes participating in the training. Length of `nodes` must be equal `number_of_nodes`.

`nodes` contains all the necessary configuration for nodes.

* `"local_epochs":` the number of local epochs to be performed on the local nodes.
* `"model_settings"` is a dictionary containing all the parameters for training the model.
  * `optimizer` is an optimizer that will be used during the training. Example: `optimizer: "RMS"`
  * `batch_size` is the batch size that will be used during the training. Example: `batch_size: 32`
  * `learning_rate` is the learning rate that will be used during the training. Example: `learning_rate: 0.001`
