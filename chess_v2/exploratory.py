#from tensorflow.keras.layers import softmax
from lib import * 


'''
def apply_temp(policy, temp):
    if temp != 0:
        scaled_policies = policy / temp
        return softmax(scaled_policies)
    else:
        return softmax(policy)
'''


def add_noise(prob, noise=0.39):
    noise = np.random.normal(0, noise, prob.shape)
    noisy_prob = prob + noise 
    noisy_prob = np.clip(noisy_prob, 0,1)
    noisy_prob /= np.sum(noisy_prob, axis=-1, keepdims=True)
    return noisy_prob

'''
model = create_model(8,256, 2)

tensor = fen_to_tensor(chess.STARTING_FEN)[np.newaxis, :]
policy, _ = model.predict(tensor)
print(policy.shape)
noisy_prob = add_noise(policy)
print(noisy_prob.shape)
'''
