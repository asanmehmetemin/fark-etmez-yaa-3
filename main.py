import torch
from dataLoader import DataLoader
from dataPreparer import prepare_dataset
from MLP import MLP
from trainer import MLPTrainer
from evaluator import Evaluator

class TxtLogger:
    def __init__(self, filename="mlp_project_results.txt"):
        self.filename = filename
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("MLP Model Experiment Results\n")
            f.write("=============================\n\n")

    def log(self, message):
        print(message)
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
def optimize_and_evaluate_mlp(dataset_name, loader, logger):
    logger.log(f"Processing Dataset: {dataset_name}")

    dataset = prepare_dataset(loader, max_features=5000)
    
    input_size = dataset["n_features"]
    output_size = dataset["n_classes"]

    logger.log("\nStep 1: Find Optimal Architecture & Regularization (For better runtime, O(n to the power of 4) is achieved instead of 6...")
    
    sample_batch = None
    sample_lr = None
    patience = None
    Archs = None
    activations = None
    batch_norms = None
    dropout_rates = None

    if dataset_name == "imdb.csv":       
        sample_batch = 64  #OPTIMISATION (pick one from Step2)
        sample_lr = 0.001  #OPTIMISATION (pick one from Steo2)
        patience = 5  #OPTIMISATION
        Archs = [(256,128),(128,64)]  #OPTIMISATION (2 li 3 lu ?)
        activations = ["relu","leaky_relu"]  #OPTIMISATION
        batch_norms = [False]  #OPTIMISATION
        dropout_rates = [0.0]  #OPTIMISATION
    elif dataset_name == "tweets.csv":
        sample_batch = 64  #OPTIMISATION (pick one from Step2)
        sample_lr = 0.001  #OPTIMISATION (pick one from Steo2)
        patience = 5  #OPTIMISATION
        Archs = [(128,64),(256,128)]  #OPTIMISATION (2 li 3 lu ?)
        activations = ["relu", "leaky_relu"]  #OPTIMISATION
        batch_norms = [False]  #OPTIMISATION
        dropout_rates = [0.3]  #OPTIMISATION

    best1_f1 = -1.0
    best1_config = None

    for archs in Archs:
        for acts in activations:
            for bn in batch_norms:
                for dropout in dropout_rates:
                    model = MLP(
                        input_size=input_size,
                        output_size=output_size,
                        hidden_layer_size=archs,
                        activation_func=acts,
                        batch_norm=bn,
                        dropout_rate=dropout
                    )

                    trainer = MLPTrainer(model=model, learning_rate= sample_lr, patience= patience)
                    model_train, _ = trainer.train_and_validate(dataset=dataset, batch_size= sample_batch, max_epochs=50,logger=logger)

                    accuracy, f1, mse, cm = Evaluator.evaluate(model_train, dataset["X_val"], dataset["Y_val"])

                    logger.log(
                        f"\nArch: {archs}, Act: {acts}, BN: {bn}, Dropout: {dropout} | "
                        f"Validation F1: {f1:.4f}"
                    )
                    if f1 > best1_f1:
                        best1_f1 = f1
                        best1_config = {
                            'hidden_layer_size': archs,
                            'activation_func': acts,
                            'batch_norm': bn,
                            'dropout_rate': dropout
                        }

    logger.log(f"\nStep 1 Best => Arch Config: {best1_config}, F1: {best1_f1:.4f}\n")

    logger.log("Step 2: Fine Tuning of batch and lr:")

    batch_sizes = None
    learning_rates = None
    patience_val = None

    if dataset_name == "imdb.csv":
        batch_sizes = [64,32]  #OPTIMISATION
        learning_rates = [0.0001,0.00007]  #OPTIMISATION
        patience_val = [2,5] #OPTIMISATION
    elif dataset_name == "tweets.csv":
        batch_sizes = [32,128]  #OPTIMISATION
        learning_rates = [0.0001,0.00005]  #OPTIMISATION
        patience_val = [5,8] #OPTIMISATION

    best_final_f1 = -1.0
    best_final_config = None
    best_model = None

    for batch in batch_sizes:
        for lr in learning_rates:
            for patience in patience_val:
                model = MLP(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_layer_size=best1_config['hidden_layer_size'],
                    activation_func=best1_config['activation_func'],
                    batch_norm=best1_config['batch_norm'],
                    dropout_rate=best1_config['dropout_rate']
                )

                trainer = MLPTrainer(model=model, learning_rate=lr, patience=patience)
                trained_model, _ = trainer.train_and_validate(
                    dataset=dataset, batch_size=batch, max_epochs=50, logger=logger
                )

                accuracy, f1, mse, cm = Evaluator.evaluate(trained_model, dataset["X_val"], dataset["Y_val"])
                logger.log(f"\nBatch: {batch}, LR: {lr}, Early Stopping Patience{patience} -> Val F1: {f1:.4f}")

                if f1 > best_final_f1:
                    best_final_f1 = f1
                    best_final_config = {
                        **best1_config,
                        'batch_size': batch,
                        'learning_rate': lr,
                        'patience': patience
                    }
                    best_model = trained_model

    logger.log(f"\nBest Overall Validation F1 Score: {best_final_f1:.4f}")
    logger.log(f"Best Configuration: {best_final_config}\n")

    #test
    logger.log(f"====Evaluating Best MLP Model on Test Set ({dataset_name})===")
    test_acc, test_f1, test_mse, test_cm = Evaluator.evaluate(best_model, dataset["X_test"], dataset["Y_test"])

    logger.log(f"Test Results for {dataset_name}:")
    logger.log(f"Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, MSE: {test_mse:.4f}")
    logger.log(f"Confusion Matrix:\n{test_cm}") 

    model_filename = f"{dataset_name.split('.')[0]}_best_mlp.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": best_final_config,
        "input_size": input_size,
        "output_size": output_size,
    }, model_filename)


if __name__ == "__main__":
    logger = TxtLogger()

    # IMDB
    optimize_and_evaluate_mlp('imdb.csv', DataLoader.load_imdb, logger)

    # Tweets
    optimize_and_evaluate_mlp('tweets.csv', DataLoader.load_tweets, logger)                         

