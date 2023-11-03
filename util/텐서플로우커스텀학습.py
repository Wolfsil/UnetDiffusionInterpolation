import tensorflow as tf
import numpy as np

def Modeling():
    inputD=tf.keras.layers.Input(shape=(3))
    inputD2=tf.keras.layers.Input(shape=(3))
    x=tf.keras.layers.Concatenate()([inputD,inputD2])
    x=tf.keras.layers.Dense(2,activation="relu")(x)
    
    return tf.keras.Model(inputs=[inputD,inputD2],outputs=[x,x])

class TestModel(tf.keras.Model):
    def __init__(self,network):
        super().__init__()
        self.network=network
        
    def train_step(self, images):

        with tf.GradientTape() as tape:
            n=self.network(images[0])
            loss=self.compiled_loss(images[1],n)
        grad=tape.gradient(loss, self.network.trainable_weights)
        print(loss)
        self.optimizer.apply_gradients(zip(grad, self.network.trainable_weights))
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, images):
        n=self.network(images[0])
        self.compiled_loss(images[1],n)
        return {m.name: m.result() for m in self.metrics}
    


model=TestModel(Modeling())

model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(
    ),
    loss=tf.keras.losses.mean_absolute_error,
    metrics=["mae"]
)
model.fit([np.random.rand(10,3),np.random.rand(10,3)],[np.random.rand(10,2),np.random.rand(10,2)],epochs=10,batch_size=2)