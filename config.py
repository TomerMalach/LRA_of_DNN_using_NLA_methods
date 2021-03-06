class config:
    batch_size = 32
    epochs = 2
    initial_learning_rate = 1e-3
    final_learning_rate = 1e-6
    lra_iterations = 100
    #kld_threshold = 0.1
    accuracy_tolerance = 0.05
    lra_model_path = 'lra_model.h5'

    @staticmethod
    def learning_rate_scheduler(e):
        return config.initial_learning_rate * (1 - e / config.epochs) + config.final_learning_rate * e / config.epochs