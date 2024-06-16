from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Flatten, Input, LeakyReLU, ELU, BatchNormalization, add, multiply, concatenate
from tensorflow.keras.models import Model

class RiceBiomassCNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.make_model()

    def make_model(self):
        inputs = Input(self.input_shape)

        x = Conv2D(45,(3,3),padding='same')(inputs)
        x = AveragePooling2D((2,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(25,(3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2,2),padding='same')(x)

        x1 = Conv2D(50,(3,3),padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D((2,3))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D((3,3),padding='same')(x1)

        x2 = Conv2D(25,(3,3),padding='same')(x)
        x2 = BatchNormalization()(x2)
        x2 = AveragePooling2D((2,3))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling2D((3,3),padding='same')(x2)

        x3 = Conv2D(16,(1,1),padding='same')(x1)        
        x3 = BatchNormalization()(x3)
        x3 = ELU(alpha=1)(x3)

        x4 = concatenate([x1,x2])
        x4 = Conv2D(16,(1,1),padding='same')(x4)        
        x4 = BatchNormalization()(x4)
        x4 = ELU(alpha=1)(x4)

        x5 = multiply([x3,x4])
        x5 = Conv2D(16,(3,3),padding='same')(x5)
        x5 = AveragePooling2D((2,2))(x5)
        x5 = BatchNormalization()(x5)
        x5 = Activation('relu')(x5)

        x6 = Conv2D(16,(3,3),padding='same')(x4)
        x6 = BatchNormalization()(x6)
        x6 = Activation('relu')(x6)
        x6 = Conv2D(16,(3,3),padding='same')(x6)
        x6 = AveragePooling2D((2,2))(x6)
        x6 = BatchNormalization()(x6)                
        
        x_m = add([x5,x6])
        x_m = Flatten()(x_m)
        predictions = Dense(1,activation='relu')(x_m)
        
        currentmodel = Model(inputs=inputs, outputs=predictions)
        
        return currentmodel

def build(input_shape=(225,300,3)):
    model = RiceBiomassCNN(input_shape).model
    return model
