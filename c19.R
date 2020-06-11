FLAGS1 <- flags(
  flag_numeric("nodes1", 128),
  flag_numeric("batch_size", 100),
  flag_string("activation", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30),
  flag_numeric("dropout", 0.5),
  flag_numeric("nodes2", 200)
)

FLAGS2 <- flags(
  flag_numeric("nodes1", 128),
  flag_numeric("batch_size", 100),
  flag_string("activation", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30),
  flag_numeric("dropout", 0.5),
  flag_numeric("nodes2", 200)
)

model = keras_model_sequential() 
model %>%
  layer_dense(units = FLAGS1$nodes1, activation = FLAGS1$activation, input_shape = dim(SECc19_train)[2]) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = FLAGS2$nodes2, activation = FLAGS2$activation) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(lr=FLAGS1$learning_rate), 
  loss = 'binary_crossentropy',
  metrics = c('accuracy'))

model %>% fit(
  SECc19_train, SECc19_labels, epochs = FLAGS1$epochs , batch_size= FLAGS1$batch_size, validation_data=list(c19_val, val_labels))