# Ageing Synthesis by CVAE / CVAE-GAN

Ageing synthesis on fundus or face images via Conditional VAE (CVAE) or CVAE-GAN model. In CVAE-GAN architecture, the training process is two-stage. You can use a pre-trained classifier or regressor to guide the generator, or you can also train the provided model.

## Data preparation

The training, validation and testing data should be saved as json format under the *./dataset* file.

> dataset
>
> > train.json
> >
> > validation.json
> >
> > test.json

A json file should contain a list of dictionaries, each of which includes at least two pieces of information: *"img_path"* and *"target"* . A example is as following:

```python
[
    {
        "img_path": "./.../1.png",
        "target": 45,
    },
    {
        "img_path": "./...2.png",
        "target": 52,
    }
]
```

where *"target"* is the training label (age in this task).

## Run

+ *cvae.py*: execute the generation task, train and test, via Conditional VAE architecture.
+ *main.py*: execute the generation task, train and test, via CVAE-GAN architecture.
+ *classification.py*: train and test a classifier.
+ *regression.py*: train and test a regressor.



