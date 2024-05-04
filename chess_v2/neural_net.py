from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, ReLU, Softmax
from tensorflow.keras.models import Model

"""
Typical Args
board_size = 8 (8x8)
num_channels = 256
Where n_layers can be increased for a deeper network

"""

def create_model(board_size, num_channels, n_layers):
    shape = (board_size, board_size, 17) # 14 = Standard chess + castling and promotion channels
    inputs = Input(shape=shape)
    
    # Convolution
    x = Conv2D(num_channels, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    


    for _ in range(n_layers):
        x = Conv2D(num_channels, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    #print(f"Addition layers, {n_layers} Successfully created")
    
    # Policy
    policy_conv = Conv2D(2, kernel_size=1)(x)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(4096, activation='softmax')(policy_flat)
    #print("Policy layers and output created")

    # Value 
    value_conv = Conv2D(1, kernel_size=1)(x)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(64, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    #print("Value layers and output created")
    model = Model(inputs=inputs, outputs=[policy_output, value_output])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
    #print("Model Created and Compiled")
    return model 


'''
model = create_model(8,256, 2)

if model is None:
    print("Model Successfully Created")
else:
    print(model.summary())
'''



