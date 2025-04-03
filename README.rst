.. _AudioProtoPNet:

============================================================
AudioProtoPNet: An Interpretable Deep Learning Model for Bird Sound Classification
============================================================

Deep learning models can recognize numerous bird species based on their vocalizations but typically function as black boxes, limiting their usefulness for domain experts.
AudioProtoPNet addresses this issue by introducing an interpretable model design based on prototype learning, allowing professionals to gain insights into the underlying computations.
Our approach demonstrates strong predictive performance on challenging multi-label bird sound classification tasks while providing valuable explanations for ornithologists and machine learning engineers.

Data
----

Our experiments are conducted on the **BirdSet** benchmark datasets, which are available on
`Hugging Face <https://huggingface.co/datasets/DBD-research-group/BirdSet>`_.
The corresponding repository, which offers additional details, is accessible on
`GitHub <https://github.com/DBD-research-group/BirdSet>`_.

Installation:
----------------------------------

All requirements are listed in ``requirements.txt``. They can be installed via:

.. code-block:: bash

    pip install -r requirements.txt

Instructions for Training the Models
------------------------------------

Train the Black-Box ConvNeXt Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, train the black-box ConvNeXt model. Navigate to the folder ``audioprotopnet/audioprotopnet/training`` and run:

.. code-block:: bash

    python train_benchmarks.py experiment="XCL/convnext/training/convnext"

This command trains a ConvNeXt model on the BirdSet XCL training dataset.

Train AudioProtoPNet
^^^^^^^^^^^^^^^^^^^^

Next, train the AudioProtoPNet model by providing the checkpoint for the previously trained ConvNeXt model.
Still in the folder ``audioprotopnet/audioprotopnet/training``, execute:

.. code-block:: bash

    python train_audioprotopnet.py experiment="XCL/audioprotopnet/training/audioprotopnet_convnext_5_prototypes" \
    module.network.model.backbone_model.local_checkpoint="path/to/convnext/checkpoint"

This step replaces the original classifier of the ConvNeXt model with our prototype learning classifier, creating an inherently interpretable model.

Instructions for Model Evaluation
---------------------------------

Evaluate the ConvNeXt Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate the ConvNeXt model on test datasets, go to ``audioprotopnet/audioprotopnet/evaluation``.
Execute the following command for each evaluation dataset (i.e., POW, PER, NES, UHH, HSN, NBP, SSW, SNE), specifying the checkpoint of the trained ConvNeXt:

.. code-block:: bash

    python eval_benchmarks.py experiment="{experiment}/convnext/inference/convnext_inference_XCL_pretrained" \
    module.network.model.local_checkpoint="path/to/convnext/checkpoint"

Evaluate the AudioProtoPNet Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate the AudioProtoPNet model on the test datasets, go to ``audioprotopnet/audioprotopnet/evaluation``.
Execute the following command for each evaluation dataset (i.e., POW, PER, NES, UHH, HSN, NBP, SSW, SNE), specifying the checkpoint of the trained AudioProtoPNet:

.. code-block:: bash

    python eval_audioprotopnet.py experiment="{experiment}/audioprotopnet/inference/audioprotopnet_convnext_inference_XCL_pretrained_5_prototypes" \
    ckpt_path="path/to/audioprotopnet/checkpoint"

Global Explanations
-------------------

Beyond quantitative performance metrics, AudioProtoPNet enables global explanations by identifying the most similar training instances to each learned prototype.
This reveals the sound patterns captured by each prototype and provides insight into what the model has learned at a global level.

To create global explanations for a specific training dataset (e.g., SNE), go to ``audioprotopnet/audioprotopnet/analysis/global_analysis`` and execute:

.. code-block:: bash

    python global_analysis.py experiment="SNE/audioprotopnet/global_analysis/global_analysis_audioprotopnet_convnext_XCL_pretrained_seed42_SNE_train" \
    ckpt_path="path/to/audioprotopnet/checkpoint"

For other datasets, adapt the configuration file accordingly.

Local Explanations
------------------

AudioProtoPNet also offers local explanations for individual instances by highlighting how the spectrogram regions of a specific recording match the learned prototypes.
This allows for deeper understanding of the model’s decision-making process case by case.

To generate local explanations for a particular test dataset (e.g., SNE), go to ``audioprotopnet/audioprotopnet/analysis/local_analysis`` and run:

.. code-block:: bash

    python local_analysis.py experiment="SNE/audioprotopnet/local_analysis/local_analysis_audioprotopnet_convnext_XCL_pretrained_seed42_SNE_test" \
    ckpt_path="path/to/audioprotopnet/checkpoint"

Again, adapt the configuration to suit your target dataset.

Authors
-------

- René Heinrich (Fraunhofer IEE, University of Kassel)
- Lukas Rauch (University of Kassel)
- Bernhard Sick (University of Kassel)
- Christoph Scholz (Fraunhofer IEE, University of Kassel)

Reference
---------

The paper for the model is available 
`here <https://www.sciencedirect.com/science/article/pii/S1574954125000901>`_.

The checkpoints for the models are available 
`here <https://huggingface.co/DBD-research-group>`_.

A tutorial on how to do inference with the models is available 
`here <https://github.com/DBD-research-group/AudioProtoPNet/blob/main/Tutorial_AudioProtoPNet.ipynb>`_.

Please cite the paper as:

.. code-block:: bibtex

    @article{heinrich2025audioprotopnet,
        title={AudioProtoPNet: An interpretable deep learning model for bird sound classification},
        author={Heinrich, Ren{\'e} and Rauch, Lukas and Sick, Bernhard and Scholz, Christoph},
        journal={Ecological Informatics},
        volume={87},
        pages={103081},
        year={2025},
        publisher={Elsevier}
    }
