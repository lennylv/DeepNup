import tensorflow as tf


def conv(input, input_size, convAFilter, convAKernel_1, convAKernel_2, ConvAstrides, prob, beta):

    x_11 = tf.keras.layers.Conv1D(convAFilter, convAKernel_1, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), strides=ConvAstrides,
                                  padding="same")(input)
    x_11 = tf.keras.layers.Activation('relu')(x_11)
    x_11 = tf.keras.layers.BatchNormalization()(x_11)
    x_11 = tf.keras.layers.MaxPool1D()(x_11)
    x_11 = tf.keras.layers.Dropout(prob)(x_11)

    x_12 = tf.keras.layers.Conv1D(convAFilter, convAKernel_1, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), strides=ConvAstrides,
                                  padding="same")(x_11)
    x_12 = tf.keras.layers.Activation('relu')(x_12)
    x_12 = tf.keras.layers.BatchNormalization()(x_12)
    x_12 = tf.keras.layers.MaxPool1D()(x_12)
    x_12 = tf.keras.layers.Dropout(prob)(x_12)

    # conv_2
    x_21 = tf.keras.layers.Conv1D(convAFilter, convAKernel_2, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), strides=ConvAstrides,
                                  padding="same")(input)
    x_21 = tf.keras.layers.Activation('relu')(x_21)
    x_21 = tf.keras.layers.BatchNormalization()(x_21)
    x_21 = tf.keras.layers.MaxPool1D()(x_21)
    x_21 = tf.keras.layers.Dropout(prob)(x_21)

    x_22 = tf.keras.layers.Conv1D(convAFilter, convAKernel_2, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), strides=ConvAstrides,
                                  padding="same")(x_21)
    x_22 = tf.keras.layers.Activation('relu')(x_22)
    x_22 = tf.keras.layers.BatchNormalization()(x_22)
    x_22 = tf.keras.layers.MaxPool1D()(x_22)
    x_22 = tf.keras.layers.Dropout(prob)(x_22)

    x = tf.keras.layers.Concatenate(1)([x_12, x_22])
    return x

def dn(input_size=(147, 4), input2_size=(64, 1),
                   convAFilter=50, convAKernel_1=5, ConvAstrides=1,  # Conv-Init
                   convAKernel_2=3,
                   unitsSize=50,
                   hidden_units=256,  # Dense Params
                   prob=0.5, learn_rate=0.0003, beta=1e-3, loss='binary_crossentropy', metrics=None):

    input1 = tf.keras.layers.Input(shape=input_size)
    input2 = tf.keras.layers.Input(shape=input2_size)

    # sequenceFeature = Encoder(1, 4, 1, 32, rate=0.5)(input1)
    conv_1 = conv(input1, input_size=input_size, convAFilter=convAFilter, convAKernel_1=convAKernel_1, convAKernel_2=convAKernel_2,
              ConvAstrides=ConvAstrides, prob=prob, beta=beta)
    conv_2 = conv(input2, input_size=input2_size, convAFilter=convAFilter, convAKernel_1=convAKernel_1, convAKernel_2=convAKernel_2,
              ConvAstrides=ConvAstrides, prob=prob, beta=beta)


    x1 = tf.keras.layers.Concatenate(1)([conv_1, conv_2])
    x1 = tf.keras.layers.MaxPool1D()(x1)

    # GRU_1
    x2 = tf.keras.layers.GRU(unitsSize, return_sequences=True)(x1)
    x2 = tf.keras.layers.MaxPool1D()(x2)
    x2 = tf.keras.layers.Dropout(prob)(x2)

    #  GRU_2
    x3 = tf.keras.layers.GRU(unitsSize, return_sequences=True)(x2)
    x3 = tf.keras.layers.MaxPool1D()(x3)
    x3 = tf.keras.layers.Dropout(prob)(x3)
    x3 = tf.keras.layers.Flatten()(x3)


    y1 = tf.keras.layers.Dense(hidden_units, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(x3)
    y1 = tf.keras.layers.Dropout(0.5)(y1)
    y1 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid')(y1)


    model = tf.keras.models.Model(inputs=[input1, input2], outputs=y1)

    # Optimizer
    optim = tf.keras.optimizers.Adam(lr=learn_rate)
    # Compile
    if (metrics != None):
        model.compile(optimizer=optim, loss=loss,
                      metrics=metrics)  # [tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
    else:
        model.compile(optimizer=optim,
                      loss=loss)  # [tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

    return model

model = dn()
tf.keras.utils.plot_model(model, to_file=r"D:\a3" + "tu" + ".png", show_shapes=True)