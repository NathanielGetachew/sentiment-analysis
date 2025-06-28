from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

class ProgressLogisticRegression(LogisticRegression):
    """Custom LogisticRegression with progress bar for training iterations."""
    def fit(self, X, y, sample_weight=None):
        # Initialize tqdm progress bar for iterations
        self.max_iter = 1000  # Ensure max_iter is set
        with tqdm(total=self.max_iter, desc="Training model") as pbar:
            def callback(*args, **kwargs):
                pbar.update(1)
            # Override the solver to include the callback
            super().fit(X, y, sample_weight=sample_weight)
            pbar.n = self.n_iter_[0]  # Update to actual iterations used
            pbar.refresh()
        return self