class Transformation(object):

    def __init__(self, pass_metrics=True, pass_training=False):
        super(Transformation, self).__init__()

        self.pass_training = pass_training
        self.pass_metrics = pass_metrics

    def __call__(self, X, **kwds):
        raise NotImplementedError