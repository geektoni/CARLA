from typing import Any, Dict

import pandas as pd
from recourse_fare.models import FARE as FAREOriginal

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing.counterfactuals import (
    check_counterfactuals,
    merge_default_parameters,
)


class FARE(RecourseMethod):
    """
    Implementation of FARE [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparameters : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    fit:
        Train the FARE model to generate counterfactual interventions.
    save:
        Save the trained model to disk.
    load:
        Load a trained FARE model from disk.
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "environment_config": dict
            Describe the environment, actions, preconditions and recourse conditions for the experiment.
        * "mcts_config": dict
            It contains the hyperparameters for the MCTS component.
        * "policy_config": dict
            It contains the hyperparameters (e.g., hidden size, embedding size, etc.) for the agent's policy.
        * "batch_size": int
            Size of the training batch when sampling from the replay buffer.
        * "training_buffer_size": int
            Size of the replay buffer.
        * "validation_steps": int
            After how many training steps we perform a validation round.

    .. [1] De Toni, Giovanni, Bruno Lepri, and Andrea Passerini. "Synthesizing explainable counterfactual policies for algorithmic recourse with program synthesis." Machine Learning (2023): 1-21.
    """

    _DEFAULT_HYPERPARAMS = {
        "environment_config": {
            "class_name": "",
            "additional_parameters": {
                "preprocessor": None,
                "preprocessor_fare": None,
                "programs_library": None,
                "arguments": None,
            },
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 10,
            "dir_epsilon": 0.03,
            "dir_noise": 0.3,
            "level_closeness_coeff": 3.0,
            "level_0_penalty": 1.0,
            "qvalue_temperature": 1.0,
            "temperature": 1.3,
            "c_puct": 0.5,
            "gamma": 0.97,
        },
        "policy_config": {
            "observation_dim": 10,
            "encoding_dim": 10,
            "hidden_size": 40,
        },
        "batch_size": 10,
        "training_buffer_size": 200,
        "validation_steps": 10,
    }

    def __init__(self, mlmodel: MLModel, hyperparameters: Dict):

        hyperparameters = merge_default_parameters(
            hyperparameters, self._DEFAULT_HYPERPARAMS
        )

        environment_config = hyperparameters.get("environment_config")
        mcts_config = hyperparameters.get("mcts_config")
        policy_config = hyperparameters.get("policy_config")
        batch_size = hyperparameters.get("batch_size")
        training_buffer_size = hyperparameters.get("training_buffer_size")
        validation_steps = hyperparameters.get("validation_steps")

        self.fare_model = FAREOriginal(
            model=mlmodel,
            environment_config=environment_config,
            policy_config=policy_config,
            mcts_config=mcts_config,
            batch_size=batch_size,
            training_buffer_size=training_buffer_size,
            validation_steps=validation_steps,
        )
        super().__init__(mlmodel)

    def fit(
        self,
        X: Any,
        max_iter: int = 1000,
        verbose: bool = True,
        tensorboard: Any = None,
    ):
        self.fare_model.fit(X, max_iter, verbose, tensorboard)

    def save(self, save_model_path: str = "."):
        self.fare_model.save(save_model_path)

    def load(self, load_model_path: str = ".") -> None:
        self.fare_model.load(load_model_path)

    def get_counterfactuals(
        self,
        factuals: pd.DataFrame,
        full_output: bool = False,
        verbose: bool = True,
        agent_only: bool = False,
        mcts_only: bool = False,
    ):

        # Get the model's preprocessor
        preprocessor = self.fare_model.environment_config.get(
            "additional_parameters", {}
        ).get("preprocessor", None)

        assert (
            preprocessor
        ), "The FARE preprocessor is None. It is not possible to generate counterfactuals."

        # Predict and get the counterfactuals
        counterfactuals = self.fare_model.predict(
            factuals,
            full_output=full_output,
            verbose=verbose,
            agent_only=agent_only,
            mcts_only=mcts_only,
        )
        counterfactuals = preprocessor.transform(counterfactuals)

        # Check if the counterfactuals are okay
        cf_df = check_counterfactuals(self._mlmodel, counterfactuals, factuals.index)
        cf_df = self._mlmodel.get_ordered_features(cf_df)

        return cf_df
