import sklearn.model_selection
# sets up from cmd:  "conda install progressbar2"
import progressbar

class GridSearchCVProgressBar(sklearn.model_selection.GridSearchCV):
    """Monkey patch Parallel to have a progress bar during grid search"""

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""

        iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)

        cv = sklearn.model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits

        class ParallelProgressBar(sklearn.model_selection._search.Parallel):
            def __call__(self, iterable):
                bar = progressbar.ProgressBar(max_value=max_value, title='GridSearchCV')
                iterable = bar(iterable)
                return super(ParallelProgressBar, self).__call__(iterable)

        # Monkey patch
        sklearn.model_selection._search.Parallel = ParallelProgressBar

        return iterator