"""
Base prior classes for Dingo.

This module provides the base DingoPrior class that other prior classes
can inherit from. It includes common functionality like the sample_as_df method.
"""

from typing import Dict, Any
import pandas as pd


class DingoPrior(dict):
    """
    Base prior class for Dingo that inherits from dict.

    This class provides a standard interface for priors used in Dingo,
    including the ability to sample parameters and return them as a pandas DataFrame.
    By inheriting from dict, subclasses automatically get dictionary methods like
    items(), __contains__, get(), keys(), etc. without having to implement them manually.
    """

    def sample(self, num_samples: int, **kwargs) -> Dict[str, Any]:
        """
        Sample parameters from the prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the prior.
        **kwargs : dict
            Additional keyword arguments passed to the underlying sample method.

        Returns
        -------
        Dict[str, Any]
            Dictionary of sampled parameters, where keys are parameter names
            and values are numpy arrays of shape (num_samples,).
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    def sample_as_df(self, num_samples: int, **kwargs) -> pd.DataFrame:
        """
        Sample parameters from the prior and return as a pandas DataFrame.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the prior.
        **kwargs : dict
            Additional keyword arguments passed to the underlying sample method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the sampled parameters, with each column
            corresponding to a parameter and each row corresponding to a sample.
        """
        samples_dict = self.sample(num_samples, **kwargs)
        return pd.DataFrame(samples_dict)
