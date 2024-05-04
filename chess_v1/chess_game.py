from self_play import *  
from data_processing import * 
from genetic_algo import * 
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard


"""
test_features, test_targets = read_pgn('20_games.pgn')
print('Successfully read in data')

test_pop = init_population(1)
test_gene = test_pop[0]
print(test_pop[0])
test_model = create_model(test_gene)
print("model created with test gene")

features_array = np.array(test_features)
targets_array = np.array(test_targets)

test_model.fit(features_array, targets_array, epochs=10, batch_size=32,validation_split=0.2)
print("Successfully Trained Model")

games, outcomes = play_self(50, test_model)
new_features, new_targets = zip(*games)
new_features = np.array(new_features)
new_targets = np.array(new_targets)

print(new_features.shape, new_targets.shape, features_array.shape, targets_array.shape)

combined_features = np.concatenate([features_array, new_features])
combined_targets = np.concatenate([targets_array, new_targets])

test_model.fit(combined_features, combined_targets, epochs = 10, batch_size=32, validation_split=0.2)

print(f"Self Play working, with {outcomes}")
"""

def saving_plots(results, gen):
    df = pd.DataFrame(results)
    df.to_csv(f'generation_{gen}_performance.csv', index = False)

    # plots
    plt.figure(figsize=(10,5))
    plt.plot(df['Generation'], df['Win Rate'], label='Win Rate')
    plt.plot(df['Generation'], df['Average Loss'], label='Average Loss')
    plt.xlabel('Generation')
    plt.ylabel('Metrics')
    plt.title('Performance Over Generations')
    plt.legend()
    plt.savefig(f'generation_{generation}_chart.png')
    plt.close()


features, targets = read_pgn('20_games.pgn')
features_array = np.array(features)
targets_array = np.array(targets)

# Initial model
gene = init_population(1)
initial_model = create_model(gene[0])
initial_model.fit(features_array, targets_array, epochs=10, batch_size=32, validation_split=0.2)
print("Initial Model Trained")
# Plotting the loss curve

# Plot histogram of the weights
weights = initial_model.get_weights()

plt.figure(figsize=(12, 6))
for i, layer_weights in enumerate(weights):
    plt.subplot(1, len(weights), i+1)
    plt.hist(layer_weights.flatten(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Layer {i+1} Weights Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("weights_histogram.png")  # Save the figure as a PNG file
plt.close()  # Close the figure to release memory


layer_name = 'activation'
conv_layer_model = Model(inputs=initial_model.input, outputs=initial_model.get_layer(layer_name).output)
conv_output = conv_layer_model.predict(features_array)
# Plotting and saving the filters visualization with color bars
plt.figure(figsize=(12, 12))
for i in range(16):  # Assuming you have 16 filters in the convolutional layer
    plt.subplot(4, 4, i+1)
    plt.imshow(conv_output[0, :, :, i], cmap='gray')
    plt.colorbar()
    plt.title(f'Filter {i+1} Activation')
    plt.axis('off')
plt.tight_layout()
plt.savefig('filters_visualization_with_colorbars.png')  # Save the figure as a PNG file
plt.close()  # Close the figure to release memory
#population = init_population(50)
#generations = 100
#results = []
""""
print("Processing Self-Play")

prev_model = initial_model
for generation in range(generations):
    log_dir = f'./logs/generation_{generation}'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    print(f"Processing Generation: {generation + 1}")
    if generation == 0:
        games, outcomes = play_self(initial_model, initial_model, n_games=5)
    else:
        games, outcomes = play_self(prev_model, best_model, n_games=5 )
        
    new_features = [game['fen'] for game in games]
    new_features_tensor = [convert_board(chess.Board(fen)) for fen in new_features]
    new_features_tensor = np.array(new_features_tensor)
    new_targets = [game['move'] for game in games]
    new_features = np.array(new_features)
    new_targets = np.array(new_targets)

    combined_features = np.concatenate([features_array, new_features_tensor], axis = 0)
    combined_targets = np.concatenate([targets_array, new_targets], axis = 0)
    data = (combined_features, combined_targets)

    best_gene = genetic_algo(data,population, population_size=50, generations=100, tournament_size=5, mut_rate=.1)
    best_model = create_model(best_gene)
    if best_model is not None:
        best_model.fit(combined_features, combined_targets, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])
        best_model.save(f'best_model_generation_{generation + 1}.h5') 
    else: 
        print("Failed to populate, skipping training")
        current_model = initial_model

    results.append({
        "Generation": generation + 1,
        "Win Rate": outcomes['win'] / 5,
        "Average Loss": np.mean(best_model.history.history['loss'])
    })
    saving_plots(results, generation + 1)
    prev_model = best_model
"""
print(f"Successfully compeleted, and saved")
