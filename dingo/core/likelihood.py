from multiprocessing import Pool

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class Likelihood(object):
    def log_likelihood(self, theta):
        raise NotImplementedError("log_likelihood() should be implemented in subclass.")

    def log_likelihood_multi(
        self, theta: pd.DataFrame, num_processes: int = 1, batch_size: int = None
    ) -> np.ndarray:
        """
        Calculate the log likelihood at multiple points in parameter space. Works with
        multiprocessing.

        This wraps the log_likelihood() method.

        Parameters
        ----------
        theta : pd.DataFrame
            Parameters values at which to evaluate likelihood.
        num_processes : int
            Number of processes to use.
        batch_size : int, optional
            Maximum number of samples to process at once. If None, processes all
            samples at once.

        Returns
        -------
        np.array of log likelihoods
        """
        with threadpool_limits(limits=1, user_api="blas"):
            if batch_size is None:
                batch_size = len(theta)

            log_likelihood = []
            for start_idx in range(0, len(theta), batch_size):
                end_idx = min(start_idx + batch_size, len(theta))
                batch = theta.iloc[start_idx:end_idx]
                batch_generator = (d[1].to_dict() for d in batch.iterrows())

                if num_processes > 1:
                    with Pool(processes=num_processes) as pool:
                        log_likelihood.extend(pool.map(self.log_likelihood, batch_generator))
                else:
                    log_likelihood.extend(map(self.log_likelihood, batch_generator))

        return np.array(log_likelihood)
